import importlib

# Example name -> (module, Problem class). Modules load only when selected,
# so the image toy example does not need ESPEI or PyCalphad.
EXAMPLES = {
    "calphad": ("examples.CALPHAD.calphad_problem", "CalphadProblem"),
    "image_toy": ("examples.image_toy.image_toy_problem", "ImageToyProblem"),
}


def load_problem(name, **kwargs):
    if name not in EXAMPLES:
        raise ValueError(f"Unknown example '{name}'. Available examples: {', '.join(EXAMPLES)}")
    module_name, class_name = EXAMPLES[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)
