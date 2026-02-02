# Post to MolHub - Instructions

This document explains how to trigger the MolHub collaboration post for the Structure-Backprop repository.

## Prerequisites

1. **MOLTHUB_API_KEY Secret**: Ensure your repository has the `MOLTHUB_API_KEY` secret configured
   - Go to: https://github.com/gnostrich/Structure-Backprop/settings/secrets/actions
   - Add a new repository secret named `MOLTHUB_API_KEY`
   - Get your API key from https://www.moltbook.com/developers

## Method 1: Trigger via GitHub UI (Easiest)

1. Go to the Actions tab: https://github.com/gnostrich/Structure-Backprop/actions
2. Click on "ClawPilot - Molthub Integration" workflow in the left sidebar
3. Click the "Run workflow" button (top right)
4. Select the branch (e.g., `main` or `copilot/collaborate-molthub-integration`)
5. Click "Run workflow"

The workflow will execute and post to MolHub community with your collaboration announcement!

## Method 2: Trigger via GitHub CLI

If you have `gh` CLI installed:

```bash
gh workflow run "ClawPilot - Molthub Integration" \
  --repo gnostrich/Structure-Backprop \
  --ref main
```

Or using the workflow file name:

```bash
gh workflow run molthub-template.yml \
  --repo gnostrich/Structure-Backprop \
  --ref main
```

## Method 3: Trigger via API

Using curl:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/gnostrich/Structure-Backprop/actions/workflows/229716855/dispatches \
  -d '{"ref":"main"}'
```

## What Will Be Posted

The workflow will post to MolHub with:

- **Title**: "🧠 Structure-First Backpropagation - New Approach to Neural Architecture Discovery"
- **Submolt**: community
- **Content**: A comprehensive post about the project including:
  - Project overview and concept
  - Key features
  - Results and achievements
  - **Call for collaboration** on various topics
  - Link to the repository

## Verify the Post

After running the workflow:

1. Check the workflow run status in GitHub Actions
2. View the logs to see the API response
3. Visit your MolHub community thread to see the post: https://www.moltbook.com/
4. Monitor for responses and collaboration offers!

## Troubleshooting

- **Workflow fails**: Check that `MOLTHUB_API_KEY` secret is configured correctly
- **No post appears**: Verify your API key is valid and has posting permissions
- **API error**: Check the workflow logs for the specific error message
