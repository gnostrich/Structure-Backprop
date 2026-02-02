# Reply to Moltbook - Instructions

This document explains how to reply to a Moltbook post directly using the GitHub Actions workflow.

## ✅ Prerequisites

The **MOLTHUB_API_KEY** secret is already configured in the repository! You're ready to reply.

## 🚀 Quick Start - Reply to a Moltbook Post

When you receive a reply on Moltbook and want to respond:

1. **Get the Post ID**: Find the ID of the Moltbook post you want to reply to
   - This is usually visible in the Moltbook URL or post metadata
   - Example: If the post URL is `https://www.moltbook.com/posts/abc123`, the post ID is `abc123`

2. **Go to the GitHub Actions page**:
   - Visit: https://github.com/gnostrich/Structure-Backprop/actions/workflows/reply-to-moltbook.yml

3. **Click "Run workflow"**:
   - Click the green **"Run workflow"** dropdown button
   - Fill in the required inputs:
     - **post_id**: Enter the ID of the post you're replying to
     - **reply_content**: Enter your reply message
   - Select the branch (e.g., `main` or current working branch)
   - Click **"Run workflow"**

That's it! Your reply will be sent to Moltbook within seconds.

## 📋 Example Usage

### Scenario: Received collaboration interest

Someone on Moltbook replied to your Structure-Backprop post showing interest in collaborating:

**Inputs:**
- **post_id**: `moltbook_post_12345`
- **reply_content**: `Thanks for your interest! We'd love to collaborate. The best place to start would be looking at the v1/example.py file which demonstrates the core concepts. Let's discuss further - feel free to open an issue on GitHub or reach out directly!`

### Scenario: Answering a technical question

Someone asked about the rounding strategy:

**Inputs:**
- **post_id**: `moltbook_post_67890`
- **reply_content**: `Great question! The rounding happens periodically during training - we snap weights to {0, 1} every N epochs. This creates a discrete structure while still allowing gradient-based learning. Check out the TRAINING_PSEUDOCODE.md in the v1 folder for the full algorithm!`

## Alternative Methods

### Method 1: Via GitHub CLI

If you have `gh` CLI installed:

```bash
gh workflow run "Reply to Moltbook Post" \
  --repo gnostrich/Structure-Backprop \
  --ref main \
  -f post_id="your_post_id_here" \
  -f reply_content="Your reply message here"
```

Or using the workflow file name:

```bash
gh workflow run reply-to-moltbook.yml \
  --repo gnostrich/Structure-Backprop \
  --ref main \
  -f post_id="your_post_id_here" \
  -f reply_content="Your reply message here"
```

### Method 2: Via GitHub API

Using curl:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/gnostrich/Structure-Backprop/actions/workflows/reply-to-moltbook.yml/dispatches \
  -d '{
    "ref":"main",
    "inputs":{
      "post_id":"your_post_id_here",
      "reply_content":"Your reply message here"
    }
  }'
```

## 🎯 Benefits of This Workflow

- **Direct reply**: No need to go through the full 'post to moltbook' flow
- **Quick response**: Reply to community feedback immediately
- **Simple**: Just provide post ID and your message
- **Tracked**: All replies are tracked in GitHub Actions logs
- **Reusable**: Use it as many times as needed for different posts

## ✅ Verify the Reply

After running the workflow:

1. Check the workflow run status in GitHub Actions
2. View the logs to see the API response
3. Visit the original Moltbook post to see your reply
4. Monitor for further engagement!

## 🔧 Troubleshooting

- **Workflow fails**: Check that `MOLTHUB_API_KEY` secret is configured correctly
- **Invalid post ID**: Verify the post ID is correct and the post exists
- **API error**: Check the workflow logs for the specific error message
- **Workflow not visible**: Make sure the workflow file is on your branch or main
- **Reply not appearing**: The post ID might be incorrect, or there might be API rate limits

## 📚 Related Documentation

- [POST_TO_MOLTHUB.md](POST_TO_MOLTHUB.md) - For creating new posts
- [.github/workflows/molthub-template.yml](.github/workflows/molthub-template.yml) - Template workflow
- [.github/workflows/reply-to-moltbook.yml](.github/workflows/reply-to-moltbook.yml) - Reply workflow
