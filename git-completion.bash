#!/bin/bash
# git-completion.bash - Auto-completion for git.sh
_git_sh_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # Main commands
    opts="init commit push branch checkout merge status history revert reset help --install-completion"
    
    # Get available branches for checkout and merge
    local branches=""
    if command -v git &> /dev/null && git rev-parse --git-dir &> /dev/null 2>&1; then
        branches=$(git branch -a 2>/dev/null | sed 's/^* //' | sed 's/remotes\/[^/]*\///' | sort -u | tr '\n' ' ')
    fi
    
    # Auto-complete based on context
    case "${prev}" in
        ./git.sh|git.sh)
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
        checkout|merge)
            if [[ -n "$branches" ]]; then
                COMPREPLY=( $(compgen -W "${branches}" -- "${cur}") )
                return 0
            fi
            ;;
        commit|push|status|history|help|init|--install-completion)
            COMPREPLY=()
            return 0
            ;;
        revert|reset|branch)
            # For these commands, don't suggest anything specific
            COMPREPLY=()
            return 0
            ;;
    esac
    
    # Default: auto-complete main commands
    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
}

complete -F _git_sh_completion ./git.sh
complete -F _git_sh_completion git.sh
