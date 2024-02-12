def get_IDs_to_labels(with_ground=False):
    IDs_to_labels = {
        0: "ABCO",
        1: "CADE",
        2: "PILA",
        3: "PIPO",
        4: "PSME",
        5: "QUEV",
        6: "SNAG",
    }
    if with_ground:
        IDs_to_labels[7] = "ground"

    return IDs_to_labels
