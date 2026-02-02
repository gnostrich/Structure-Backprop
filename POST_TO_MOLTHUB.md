# Post to MolHub - Instructions

This document explains how to trigger the MolHub collaboration post for the Structure-Backprop repository.

## ✅ Prerequisites

The **MOLTHUB_API_KEY** secret is already configured in the repository! You're ready to post.

## 🚀 Quick Start - Post Now!

**RECOMMENDED:** Use the dedicated "Post to MolHub" workflow:

1. Go to: https://github.com/gnostrich/Structure-Backprop/actions/workflows/post-to-molthub.yml
2. Click the green **"Run workflow"** button
3. Select the branch (e.g., `copilot/collaborate-molthub-integration` or `main`)
4. Click **"Run workflow"**

That's it! The post will be sent to MolHub within seconds.

## Alternative Methods

### Method 1: Via GitHub Actions UI (Original Workflow)

Use the original template workflow:

1. Go to the Actions tab: https://github.com/gnostrich/Structure-Backprop/actions
2. Click on "ClawPilot - Molthub Integration" workflow in the left sidebar
3. Click the "Run workflow" button (top right)
4. Select the branch
5. Click "Run workflow"

### Method 2: Trigger via GitHub CLI

If you have `gh` CLI installed, use the new dedicated workflow:

```bash
gh workflow run "Post to MolHub" \
  --repo gnostrich/Structure-Backprop \
  --ref copilot/collaborate-molthub-integration
```

Or using the workflow file name:

```bash
gh workflow run post-to-molthub.yml \
  --repo gnostrich/Structure-Backprop \
  --ref copilot/collaborate-molthub-integration
```

### Method 3: Trigger via API

Using curl with the new dedicated workflow:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/gnostrich/Structure-Backprop/actions/workflows/post-to-molthub.yml/dispatches \
  -d '{"ref":"copilot/collaborate-molthub-integration"}'
```

## 📋 What Will Be Posted

The workflow will post to MolHub with:

- **Title**: "🧠 Structure-First Backpropagation - New Approach to Neural Architecture Discovery"
- **Submolt**: community
- **Content**: A comprehensive post about the project including:
  - Project overview and concept
  - Key features (dense graph, gradient learning, structure discovery)
  - Results (45-53% sparsity, working examples)
  - **Call for collaboration** on:
    - Novel applications of structure learning
    - Extensions to different problem domains
    - Improved rounding/pruning strategies
    - Connections to other graph learning methods
    - Theoretical analysis and convergence properties
  - Link to the repository

## ✅ Verify the Post

After running the workflow:

1. Check the workflow run status in GitHub Actions
2. View the logs to see the API response
3. Visit your MolHub community thread: https://www.moltbook.com/
4. Monitor for responses and collaboration offers!

## 🔧 Troubleshooting

- **Workflow fails**: Check that `MOLTHUB_API_KEY` secret is configured correctly
- **No post appears**: Verify your API key is valid and has posting permissions
- **API error**: Check the workflow logs for the specific error message
- **Workflow not visible**: Make sure the workflow file is on your branch or main
