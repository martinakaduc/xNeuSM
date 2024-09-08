import os

dirs = os.listdir("./raw_datasets")

for dirs in [os.path.join("./raw_datasets", d) for d in dirs]:
    files = os.listdir(dirs)
    # Get file ending with .node_labels
    file = [f for f in files if f.endswith(".node_labels")][0]
    with open(dirs + "/" + file, "r") as f:
        lines = f.readlines()
    lines = [int(line) for line in lines if line != ""]
    labels = set(lines)
    print("Number of nodes in " + dirs + ": " + str(len(labels)) + " nodes.")

    if "DBLP-v1" in dirs:
        continue
    counts = {x: lines.count(x) for x in labels}
    # Find min count and index
    min_count = min(counts.values())
    min_index = {k: v for k, v in counts.items() if v == min_count}
    min_index = list(min_index.keys())[0]
    print(
        "Number of nodes: "
        + str(min_count)
        + " for source graph #"
        + str(min_index)
        + "."
    )
