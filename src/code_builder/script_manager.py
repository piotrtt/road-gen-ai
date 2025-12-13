class ScriptBuilder:
    """
    Manages the assembly of the final Python script that uses 'scenariogeneration'.
    
    Responsible for:
    - Managing imports
    - Tracking variable names for road segments to link predecessors and successors
    - Assembling the sequential code blocks
    """
    
    def __init__(self):
        self.code_blocks = []
        self.imports = []
        self.variables = {} # Map 'id' -> 'var_name'

    def init_script(self):
        """Initializes the script with necessary imports and setup."""
        self.imports.append("from scenariogeneration import xodr")
        # TODO: Add other standard setup code

    def add_component(self, code: str):
        """
        Adds a generated code block for a component to the script.
        
        Args:
            code (str): The python code string for creating a road component.
        """
        self.code_blocks.append(code)
        # TODO: Need to handle variable assignment here, e.g., "road1 = ..."
        # and linking to previous roads if necessary.

    def get_current_script(self) -> str:
        """
        Returns the complete Python script as a string.
        """
        script = "\n".join(self.imports) + "\n\n"
        script += "\n".join(self.code_blocks)
        return script
