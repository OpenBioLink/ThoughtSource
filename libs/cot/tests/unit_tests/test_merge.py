from .utils import chdir, json_file_contains, get_test_collection
from cot import Collection

def test_child_in_merged_file() -> None:
    """Test that one of the children files is in the merged parent."""
    
    #child 1 is a subset of child 2
    child_1_file_path =  "worldtree_1_merge"
    child_2_file_path = "worldtree_2_merge"

    #intend to merge child 1 and 2
    merge_col1 = get_test_collection(child_1_file_path)
    merge_col2 = get_test_collection(child_2_file_path)

    merge = merge_col1.merge(merge_col2)

    with chdir("unit_tests/data"):
        merge.dump('childs_merged')

    childs_merged = 'childs_merged'
    
    """Assert children are subset of merged file"""
    assert json_file_contains(child_1_file_path, childs_merged) == True    
    assert json_file_contains(child_2_file_path, childs_merged) == True
    




