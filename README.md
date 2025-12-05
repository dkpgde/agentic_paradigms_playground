4 tools in all scenarios.    
Interested in agents for on-device use cases. Results don't generalize for bigger models/toolsets.   
For this simplistic use, clearly the complexity of hierarchical agents or code mode MCP is counterproductive.  
Similarly, a larger model is also counterproductive, and/or requires a different system prompt/tool description.   
However, even on this simple taks with miniscule tool call results, the lower token usage by Granite-tiny when using code is visible.  
Performance likely explained by normal MCP beuing easier for a 7B model.

Code mode MCP:  
[Granite4:tiny-h by IBM](https://huggingface.co/collections/ibm-granite/granite-40-language-models): 68% success rate; 515 seconds; 38k tokens.
[Qwen 2.5 14B](https://huggingface.co/collections/Qwen/qwen25): 30% success rate out of finished tasks; 55% of tasks resulted in crashes due to exceeding 300 second per-task timeout; 97.5k tokens in 4081 seconds. 

Default MCP:  
[Granite4:tiny-h by IBM](https://huggingface.co/collections/ibm-granite/granite-40-language-models): 91% success rate; 439 seconds; 46k tokens. 
[Qwen 2.5 14B](https://huggingface.co/collections/Qwen/qwen25): 84% success rate; 1216 seconds; 59k tokens.  

Orchestrator + 2 workers:
Using [granite4:tiny-h by IBM](https://huggingface.co/collections/ibm-granite/granite-40-language-models).
Success rate 59%; dragged down by longer tasks.

