from typing import Optional, List
import json
import re
from collections import deque # Import deque for tree_node helper

# --- START: Add Actual Class Definitions for Type Hinting ---
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# --- END: Add Actual Class Definitions for Type Hinting ---

# Standard class definitions (as strings for rationale)
LIST_NODE_DEF = """\
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""

TREE_NODE_DEF = """\
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""

# --- START: Define Standard Helper Functions --- 
LIST_NODE_HELPER_DEF = """\
# Helper function to create ListNode from list
def list_node(vals):
    if not vals:
        return None
    head = ListNode(vals[0])
    current = head
    for i in range(1, len(vals)):
        current.next = ListNode(vals[i])
        current = current.next
    return head
"""

TREE_NODE_HELPER_DEF = """\
# Helper function to create TreeNode from list (level order)
from collections import deque
def tree_node(vals):
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    queue = deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            queue.append(node.right)
        i += 1
    return root
"""

# --- START: Define is_same_list helper string --- 
IS_SAME_LIST_FUNC_STR = """\
# Helper function to compare two linked lists
def is_same_list(l1: Optional[ListNode], l2: Optional[ListNode]) -> bool:
    while l1 and l2:
        if l1.val != l2.val:
            return False
        l1 = l1.next
        l2 = l2.next
    return l1 is None and l2 is None
"""
# --- END: Define is_same_list helper string --- 

# --- START: Define is_same_tree helper string --- 
IS_SAME_TREE_FUNC_STR = """\
# Helper function to compare two trees
def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""
# --- END: Define is_same_tree helper string --- 

# --- END: Define Standard Helper Functions --- 

# Helper function to compare two linked lists (example)
def is_same_list(l1: Optional[ListNode], l2: Optional[ListNode]) -> bool:
    """Returns True if the two linked lists l1 and l2 contain the same sequence of values and are the same length; otherwise returns False."""
    # Definition assumes ListNode class is available in the execution context
    p, q = l1, l2
    while p is not None and q is not None:
        if p.val != q.val:
            return False
        p = p.next
        q = q.next
    # Both should reach the end together
    return p is None and q is None

# Define input and output file paths
input_file_path = 'loong/data/programming/submit_result.json'
output_file_path = 'loong/data/programming/seed_dataset.json'

# Helper function to extract signature and method details
def extract_signature_details(signature_block):
    """Extracts the first method name and parameters after 'class Solution'."""
    # Find the first method definition after 'class Solution'
    # Adjusted regex to be more robust and handle different spacings/newlines
    # Modified regex to capture parameters up to the closing parenthesis, making the colon optional
    method_match = re.search(r'class Solution.*?\n\s*def\s+(\w+)\s*\((.*?)\)', signature_block, re.DOTALL | re.IGNORECASE)
    if method_match:
        method_name = method_match.group(1)
        params_with_self = method_match.group(2).strip()
        # Remove 'self', handle potential type hints, and extract param names
        params_list = []
        # Simple split first, then refine to get just names
        raw_params = [p.strip() for p in params_with_self.split(',')]
        if raw_params and raw_params[0].startswith('self'):
            raw_params.pop(0) # Remove self

        # Extract just the names (part before ':')
        for p in raw_params:
             param_name = p.split(':')[0].strip()
             if param_name: # Ensure it's not empty
                 params_list.append(param_name)

        params_names_only = ', '.join(params_list)
        full_params_without_self = ', '.join(raw_params) # Keep original types for candidate signature

        return method_name, full_params_without_self, params_names_only
    return None, None, None


# Load the input JSON data
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_file_path}")
    exit(1)

processed_data = []

# Process each entry in the JSON data (assuming it's a list)
if not isinstance(data, list):
    print("Error: Expected input JSON to be a list of objects.")
    exit(1)

for entry in data:
    question = entry.get('question', '')
    original_rationale = entry.get('rationale', None) # Use None if not present
    answer = entry.get('answer', '') # This will become metadata['reasoning']

    # 1. Extract PROVIDED_CODE
    provided_code = ""
    code_marker = '[Code]'
    code_start_index = question.find(code_marker)
    if code_start_index != -1:
        provided_code = question[code_start_index + len(code_marker):].strip()
        # Remove the specific unwanted sentence
        sentence_to_remove = "You will always receive the input in this format. Do not output anything other than the final object."
        if provided_code.endswith(sentence_to_remove):
             provided_code = provided_code[:-len(sentence_to_remove)].strip()


    # 2. Extract SIGNATURE
    signature_block = ""
    signature_match = re.search(r'```python\n(.*?)```', question, re.DOTALL)
    if signature_match:
        signature_block = signature_match.group(1).strip()


    # 3. Extract Method Name and Parameters from SIGNATURE
    method_name, full_params_without_self, params_names_only = extract_signature_details(signature_block)

    # 4. Construct CANDIDATE_FUNC
    candidate_func_str = ""
    typing_import_needed = False
    if method_name and full_params_without_self is not None:
        # Construct the call part using only param names
        call_params = params_names_only if params_names_only else ""
        # Use full params (with types) for the candidate signature
        candidate_signature_params = full_params_without_self if full_params_without_self else ""

        # Extract parameters without 'self' but with types for the def line
        params_parts = [p.strip() for p in full_params_without_self.split(',')]
        params_for_def_list = [p for p in params_parts if p != 'self']
        params_for_def = ", ".join(params_for_def_list)

        # Extract just the names without 'self' for the call line
        param_names_for_call_list = [p.split(':')[0].strip() for p in params_for_def_list]
        call_params = ", ".join(param_names_for_call_list)

        # Construct the candidate function string
        candidate_func_str = f"""
def candidate({params_for_def}):
    solution_instance = Solution()
    return solution_instance.{method_name}({call_params})
"""

        # Check if imports are needed for the candidate signature
        if 'Optional' in candidate_signature_params or 'ListNode' in candidate_signature_params or 'List' in candidate_signature_params:
             typing_import_needed = True


    # 5. Construct the new rationale string
    new_rationale_parts = []
    imports_needed = set() # Store necessary imports
    class_defs_needed = set() # Store necessary class definitions
    helper_defs_needed = set() # Store needed standard helper defs

    # Combine signature parts for checking type usage
    full_signature_text = signature_block # Use signature block as primary source
    solution_code_block = entry.get('solution_code_block', '').strip()

    if 'Optional[' in full_signature_text:
        imports_needed.add("from typing import Optional")
    if 'List[' in full_signature_text:
        imports_needed.add("from typing import List")
    if 'ListNode' in full_signature_text:
        class_defs_needed.add(LIST_NODE_DEF)
        helper_defs_needed.add(LIST_NODE_HELPER_DEF)
        helper_defs_needed.add(IS_SAME_LIST_FUNC_STR) # Add is_same_list if ListNode is used
    if 'TreeNode' in full_signature_text:
         class_defs_needed.add(TREE_NODE_DEF)
         helper_defs_needed.add(TREE_NODE_HELPER_DEF)
         helper_defs_needed.add(IS_SAME_TREE_FUNC_STR) # Add is_same_tree if TreeNode is used
         imports_needed.add("from typing import Optional") # is_same_tree requires Optional
         # TreeNode helper needs deque
         imports_needed.add("from collections import deque")


    # Check for 'collections.' usage OR standalone 'deque('
    imports_needed.add("import math")
    # Deque import might already be added if TreeNode is used, but check solution code too.
    if re.search(r'collections\.', solution_code_block) or re.search(r'\bdeque\(', solution_code_block):
        # Prefer 'import collections' for simplicity if direct usage or deque() is found.
        # Avoid adding if 'from collections import deque' is already there from TreeNode handling.
        if "from collections import deque" not in imports_needed:
            imports_needed.add("import collections")
        # Ensure deque itself is available if 'from collections import deque' was added for TreeNode
        # but solution code *also* uses collections.deque (this adds import collections too)
        elif re.search(r'collections\.deque', solution_code_block):
            imports_needed.add("import collections")

    # Ensure 'deque' import specifically if TreeNode used it, even if we added 'import collections'
    # This covers the case where solution code uses 'collections.' but not 'deque', yet TreeNode needs it.
    if 'TreeNode' in full_signature_text:
        # We now add 'import collections' above if needed.
        # Ensure 'from collections import deque' is still added specifically for tree_node helper.
        imports_needed.add("from collections import deque")
        imports_needed.add("import collections")

    # --- Assemble Rationale Parts ---
    imports_str = "\n".join(sorted(list(imports_needed)))
    if imports_str:
        new_rationale_parts.append(imports_str)
    class_defs_str = "\n\n".join(class_defs_needed)
    if class_defs_str:
        new_rationale_parts.append(class_defs_str) # Add class defs here

    if provided_code:
        # Ensure PROVIDED_CODE doesn't duplicate the definitions we just added
        # This might be tricky, but let's assume PROVIDED_CODE has the Solution class
        new_rationale_parts.append(provided_code)

    if original_rationale:
        new_rationale_parts.append(original_rationale) # Append original rationale content

    # Ensure candidate function is added if extracted
    if candidate_func_str:
        new_rationale_parts.append(candidate_func_str)

    # Add standard helper definitions
    helper_defs_str = "\n\n".join(helper_defs_needed)
    if helper_defs_str:
        new_rationale_parts.append(helper_defs_str)

    # 7. Add check(candidate) call
    check_call_str = "check(candidate)" 
    # Add check function definition if not already present (assuming check is defined globally or provided)
    # For simplicity, we just add the call. Ensure 'check' is defined in the execution environment.
    new_rationale_parts.append(check_call_str)

    # 7. Add print True
    new_rationale_parts.append("\n\nprint(True)")

    new_rationale = "\n\n".join(new_rationale_parts).strip() # Join parts with double newline and strip leading/trailing whitespace

    # Clean up potentially remaining triple backticks or stray sentences
    new_rationale = new_rationale.replace(r"```\n\nYou will always receive the input in this format. Do not output anything other than the final object.", "")
    new_rationale = new_rationale.replace("```", "")

    # 7. Create the new structure
    new_entry = entry.copy() # Start with a copy of the original entry
    new_entry['rationale'] = new_rationale
    if 'answer' in new_entry:
        del new_entry['answer'] # Remove original answer field
    if 'requirements' in new_entry:
        del new_entry['requirements'] # Remove original requirements field
    new_entry['metadata'] = new_entry.get('metadata', {})
    new_entry['metadata']['reasoning'] = answer
    new_entry['metadata']['required_dependencies'] = []
    new_entry['final_answer'] = "True"

    processed_data.append(new_entry)

# Write the processed data to the output JSON file
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)
    print(f"Successfully processed data and saved to {output_file_path}")
except IOError as e:
    print(f"Error writing to output file {output_file_path}: {e}")
