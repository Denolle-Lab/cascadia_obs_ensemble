# Reverse sync: Overleaf edits → GitHub issues (install on the PAPER repo)

`overleaf-edit-to-issue.yml` belongs on **Denolle-Lab/Offshore-Cascadia-Ensemble-Catalog**
(the Overleaf-linked repo), NOT here. It detects direct Overleaf-editor edits (any push
to that repo not authored by `cascadia-sync[bot]`) and files an issue here with the diff,
so the edit gets ported into `paper/main.qmd` instead of being overwritten by the next
forward sync.

It is kept in this repo (rather than pushed automatically) because creating
`.github/workflows/*` files over the API/CLI requires a `workflow`-scoped token; add it
by hand once:

```sh
# from a checkout of the paper repo
mkdir -p .github/workflows
cp .../paper/overleaf-sync/overleaf-edit-to-issue.yml .github/workflows/
git add .github/workflows/overleaf-edit-to-issue.yml
git commit -m "Add reverse Overleaf-edit -> issue workflow" && git push
```

Then set its secret + label (see the workflow header):
```sh
# CASCADIA_ISSUES_TOKEN = a fine-grained PAT (Issues:write on cascadia_obs_ensemble),
# created via the GitHub web UI, stored on the PAPER repo:
gh secret set CASCADIA_ISSUES_TOKEN --repo Denolle-Lab/Offshore-Cascadia-Ensemble-Catalog < token.txt
gh label create overleaf-sync --repo Denolle-Lab/cascadia_obs_ensemble -c FBCA04
```
