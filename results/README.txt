
Directory structure:

(datasets taken from miplib2010)


[dataset] 
    [sb-0] // 100% strong branching, theoretical best (actually? not sure)
    [sb-#-pc] // Strong branching for # iterations followed by pseudocost branching (our benchmark to beat) (not sure how to get this yet)
    [sb-#-nn] // String brancking for # iterations followed by our NN (this is working)
        [#1_#2_...] // size and # of layers
            [#] // # of epochs trained
                [?] // Anything else??? can do activation funcs, features, etc tho may want to rearrange orders for those        





At what level should we switch up CPLEX random seeds?? should we ever switch up torch random seeds?



