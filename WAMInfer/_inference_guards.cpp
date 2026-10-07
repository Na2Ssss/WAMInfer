#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include <vector>

namespace {
struct TensorState {
    at::Tensor value;
    const void* data;
    std::vector<int64_t> sizes, strides;
    c10::Device device;
    at::ScalarType dtype;
    uint32_t version;
    bool versioned;

    explicit TensorState(const at::Tensor& x)
        : value(x), data(x.data_ptr()), sizes(x.sizes().vec()), strides(x.strides().vec()),
          device(x.device()), dtype(x.scalar_type()), version(0),
          versioned(x.unsafeGetTensorImpl()->version_counter().enabled()) {
        if (versioned) version = x.unsafeGetTensorImpl()->version_counter().current_version();
    }
    bool matches(bool track_version) const {
        if (value.data_ptr() != data || value.device() != device || value.scalar_type() != dtype ||
            value.sizes() != at::IntArrayRef(sizes) || value.strides() != at::IntArrayRef(strides)) return false;
        const auto& counter = value.unsafeGetTensorImpl()->version_counter();
        return !track_version || (versioned && counter.enabled() && counter.current_version() == version);
    }
};

struct DictionaryState {
    pybind11::dict value;
    std::vector<std::pair<pybind11::object, pybind11::object>> items;
    explicit DictionaryState(pybind11::dict d) : value(std::move(d)) {
        Py_ssize_t pos = 0; PyObject *key, *item;
        while (PyDict_Next(value.ptr(), &pos, &key, &item)) items.emplace_back(
            pybind11::reinterpret_borrow<pybind11::object>(key),
            pybind11::reinterpret_borrow<pybind11::object>(item));
    }
    bool matches(const pybind11::object& current) const {
        if (current.ptr() != value.ptr() || PyDict_Size(value.ptr()) != Py_ssize_t(items.size())) return false;
        Py_ssize_t pos = 0; PyObject *key, *item; size_t index = 0;
        while (PyDict_Next(value.ptr(), &pos, &key, &item)) {
            const auto& saved = items[index++];
            if (saved.first.ptr() != key || saved.second.ptr() != item) return false;
        }
        return true;
    }
};

struct AttributeNames {
    pybind11::str children{"_modules"}, parameters{"_parameters"}, buffers{"_buffers"}, training{"training"};
};

struct ModuleState {
    pybind11::weakref module;
    DictionaryState children, parameters, buffers;
    ModuleState(const pybind11::object& m, const AttributeNames& names)
        : module(m), children(m.attr(names.children)), parameters(m.attr(names.parameters)), buffers(m.attr(names.buffers)) {}
    bool matches(const AttributeNames& names) const {
        auto m = pybind11::reinterpret_borrow<pybind11::object>(PyWeakref_GetObject(module.ptr()));
        return !m.is_none() && children.matches(m.attr(names.children)) &&
            parameters.matches(m.attr(names.parameters)) && buffers.matches(m.attr(names.buffers));
    }
};

class ModuleGuard {
    pybind11::weakref root_;
    bool track_versions_;
    AttributeNames names_;
    std::vector<ModuleState> modules_;
    std::vector<TensorState> tensors_;

    void walk(const pybind11::object& m, std::unordered_set<PyObject*>& seen_modules,
              std::unordered_set<const void*>& seen_tensors) {
        if (!seen_modules.insert(m.ptr()).second) return;
        modules_.emplace_back(m, names_);
        for (const auto& name : {names_.parameters, names_.buffers}) {
            auto values = pybind11::reinterpret_borrow<pybind11::dict>(m.attr(name));
            for (auto item : values) {
                if (THPVariable_Check(item.second.ptr())) {
                    const auto& x = THPVariable_Unpack(item.second.ptr());
                    if (seen_tensors.insert(x.unsafeGetTensorImpl()).second) tensors_.emplace_back(x);
                }
            }
        }
        auto children = pybind11::reinterpret_borrow<pybind11::dict>(m.attr(names_.children));
        for (auto item : children) if (!item.second.is_none())
            walk(pybind11::reinterpret_borrow<pybind11::object>(item.second), seen_modules, seen_tensors);
    }
    void snapshot() {
        auto root = root_();
        TORCH_CHECK(!root.is_none(), "Inference guard's model was released");
        modules_.clear(); tensors_.clear();
        std::unordered_set<PyObject*> modules;
        std::unordered_set<const void*> tensors;
        walk(root, modules, tensors);
    }
public:
    ModuleGuard(pybind11::object root, bool track_versions) : root_(root), track_versions_(track_versions) { snapshot(); }
    bool all_eval() const {
        // The root flag alone misses a child explicitly switched to train().
        // A replaced/added child invalidates this topology snapshot too.
        for (const auto& state : modules_) {
            auto module = pybind11::reinterpret_borrow<pybind11::object>(PyWeakref_GetObject(state.module.ptr()));
            if (module.is_none() || module.attr(names_.training).ptr() != Py_False ||
                !state.children.matches(module.attr(names_.children))) return false;
        }
        return true;
    }
    bool changed() {
        for (const auto& module : modules_) if (!module.matches(names_)) { snapshot(); return true; }
        for (const auto& tensor : tensors_) if (!tensor.matches(track_versions_)) { snapshot(); return true; }
        return false;
    }
};
// Validate hot Graph input layouts without rebuilding Python pytree metadata.
struct InputNode {
    enum Kind { Tuple, List, Dict, Tensor, Scalar } kind;
    std::vector<InputNode> children;
    std::vector<pybind11::object> keys;
    pybind11::object scalar;
    std::vector<int64_t> sizes, strides;
    c10::Device device{c10::DeviceType::CPU};
    at::ScalarType dtype = at::kFloat;
    bool supported = true;

    explicit InputNode(pybind11::handle value) {
        if (PyTuple_CheckExact(value.ptr()) || PyList_CheckExact(value.ptr())) {
            kind = PyTuple_CheckExact(value.ptr()) ? Tuple : List;
            auto count = PySequence_Size(value.ptr());
            children.reserve(count);
            for (Py_ssize_t i = 0; i < count; ++i) {
                auto item = kind == Tuple ? PyTuple_GET_ITEM(value.ptr(), i) : PyList_GET_ITEM(value.ptr(), i);
                children.emplace_back(pybind11::handle(item));
                supported &= children.back().supported;
            }
        } else if (PyDict_CheckExact(value.ptr())) {
            kind = Dict;
            Py_ssize_t pos = 0; PyObject *key, *item;
            children.reserve(PyDict_Size(value.ptr()));
            while (PyDict_Next(value.ptr(), &pos, &key, &item)) {
                keys.emplace_back(pybind11::reinterpret_borrow<pybind11::object>(key));
                children.emplace_back(pybind11::handle(item));
                supported &= children.back().supported;
            }
        } else if (THPVariable_Check(value.ptr())) {
            kind = Tensor;
            const auto& x = THPVariable_Unpack(value.ptr());
            sizes = x.sizes().vec(); strides = x.strides().vec(); device = x.device(); dtype = x.scalar_type();
        } else {
            kind = Scalar;
            supported = value.is_none() || PyBool_Check(value.ptr()) || PyLong_CheckExact(value.ptr()) ||
                PyFloat_CheckExact(value.ptr()) || PyUnicode_CheckExact(value.ptr());
            scalar = pybind11::reinterpret_borrow<pybind11::object>(value);
        }
    }
    bool flatten(pybind11::handle value, pybind11::list& leaves) const {
        switch (kind) {
        case Tuple: case List: {
            if ((kind == Tuple && !PyTuple_CheckExact(value.ptr())) ||
                (kind == List && !PyList_CheckExact(value.ptr())) ||
                PySequence_Size(value.ptr()) != Py_ssize_t(children.size())) return false;
            for (size_t i = 0; i < children.size(); ++i) {
                auto item = kind == Tuple ? PyTuple_GET_ITEM(value.ptr(), i) : PyList_GET_ITEM(value.ptr(), i);
                if (!children[i].flatten(pybind11::handle(item), leaves)) return false;
            }
            return true;
        }
        case Dict: {
            if (!PyDict_CheckExact(value.ptr()) || PyDict_Size(value.ptr()) != Py_ssize_t(children.size())) return false;
            Py_ssize_t pos = 0; PyObject *key, *item; size_t index = 0;
            while (PyDict_Next(value.ptr(), &pos, &key, &item)) {
                if (PyObject_RichCompareBool(keys[index].ptr(), key, Py_EQ) != 1 ||
                    !children[index].flatten(pybind11::handle(item), leaves)) return false;
                ++index;
            }
            return true;
        }
        case Tensor: {
            if (!THPVariable_Check(value.ptr())) return false;
            const auto& x = THPVariable_Unpack(value.ptr());
            if (x.device() != device || x.scalar_type() != dtype || x.sizes() != at::IntArrayRef(sizes) ||
                x.strides() != at::IntArrayRef(strides)) return false;
            leaves.append(value); return true;
        }
        case Scalar:
            if (Py_TYPE(value.ptr()) != Py_TYPE(scalar.ptr()) || PyObject_RichCompareBool(value.ptr(), scalar.ptr(), Py_EQ) != 1)
                return false;
            leaves.append(value); return true;
        }
        return false;
    }
};

class InputGuard {
    InputNode root_;
public:
    explicit InputGuard(pybind11::object value) : root_(value) {}
    pybind11::object flatten(pybind11::object value) const {
        pybind11::list leaves;
        if (!root_.supported || !root_.flatten(value, leaves)) return pybind11::none();
        return leaves;
    }
};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<ModuleGuard>(module, "ModuleGuard")
        .def(pybind11::init<pybind11::object, bool>()).def("changed", &ModuleGuard::changed)
        .def("all_eval", &ModuleGuard::all_eval);
    pybind11::class_<InputGuard>(module, "InputGuard")
        .def(pybind11::init<pybind11::object>()).def("flatten", &InputGuard::flatten);
}
