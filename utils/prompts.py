SYSTEM_PROMPT = """ # INSTRUCTION 
I will provide you a structure like this with question and evidences filled. Your response should only contain the new structure with filled answer to the question and thats it. ONLY USE THE EXACT ANSWER, NO EXPLANATIONS. Don't generate anything more. Please return a structure with the final answer filled based on the provided question and evidences.
"""

EXAMPLE_USER_PROMPT = """ "question": "What is the sports team the person played for who scored the first touchdown in Superbowl 1?",
    "evidences": [
        "The player that scored the first touchdown in Superbowl 1 is Max McGee.",
        "Max McGee played for the Green Bay Packers."
      ],
"""

EXAMPLE_ASSISTANT_RESPONSE = """{"final_answer": "Green Bay Packers."}"""
