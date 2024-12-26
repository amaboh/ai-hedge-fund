from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage


import json


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]



def show_agent_reasoning(output, agent_name):
    """Display agent reasoning with enhanced error handling and debugging."""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    
    
    def convert_to_serializable(obj):
        if hasattr(obj, 'to_dict'):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation
    
    try:
        # If output is already a dict or list, convert and print
        if isinstance(output, (dict, list)):
            serializable_output = convert_to_serializable(output)
            print(json.dumps(serializable_output, indent=2))
        else:
            # Try to parse string as JSON
            try:
                parsed_output = json.loads(output)
                print(json.dumps(parsed_output, indent=2))
            except json.JSONDecodeError as e:
                print(f"\nDebug: JSON parsing error: {str(e)}")
                print("Falling back to original string output:")
                print(output)
    except Exception as e:
        print("Original output:")
        print(output)
    
    print("=" * 48)