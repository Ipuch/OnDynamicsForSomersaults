import biorbd
model_path = "../Model_JeCh_10DoFs.bioMod"
m = biorbd.Model(model_path)
children = m.GetChildRbdlIdx()

end_branch = []
for i in range(len(children)):
    child = children[i]
    if len(child) == 0:
        end_branch.append(i)

subtree = []


# begin where solids as no parents
# subtree = [];
# for all solid with no parent
#   do subtree = sub_tree(subtree, i)

def recursive_children(subtree):
    c = children[subtree[-1]]
    print(c)
    if c:
        for i in range(len(c)):
            subtree.append(c[i])
            print(subtree.sort())
            recursive_children(subtree)

    return (subtree)
i=0
subtree = []
subtree.append(i)
recursive_children(subtree)
print(subtree.sort())