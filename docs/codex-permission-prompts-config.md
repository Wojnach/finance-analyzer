Configure Codex to reduce command permission prompts for my development workflow.  
  
1. Create directory if it does not exist:  
  
~/.codex/rules  
  
2. Create the file:  
  
~/.codex/rules/default.rules  
  
3. Add the following rules:  
  
prefix_rule(pattern=["git"], decision="allow")  
prefix_rule(pattern=["ls"], decision="allow")  
prefix_rule(pattern=["cat"], decision="allow")  
prefix_rule(pattern=["rg"], decision="allow")  
prefix_rule(pattern=["grep"], decision="allow")  
  
prefix_rule(pattern=["python"], decision="allow")  
prefix_rule(pattern=["pytest"], decision="allow")  
  
prefix_rule(pattern=["mkdir"], decision="allow")  
prefix_rule(pattern=["cp"], decision="allow")  
prefix_rule(pattern=["mv"], decision="allow")  
  
prefix_rule(pattern=["tmux"], decision="allow")  
prefix_rule(pattern=["htop"], decision="allow")  
  
4. Ensure a user config exists at:  
  
~/.codex/config.toml  
  
If the file does not exist, create it with:  
  
approval_policy = "on-request"  
sandbox_mode = "workspace-write"  
  
5. Do NOT modify anything outside the ~/.codex directory.  
  
6. After finishing, print the file paths and contents.  
