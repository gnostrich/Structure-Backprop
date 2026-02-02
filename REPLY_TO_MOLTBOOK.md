# Reply to Moltbook - Instructions

This document explains how to reply to posts on Moltbook using the GitHub Actions workflow.

## ✅ Prerequisites

The **MOLTHUB_API_KEY** secret is already configured in the repository! You're ready to reply.

## 🚀 Quick Start - Reply Now!

**Use the "Reply to Moltbook Post" workflow:**

1. Go to: https://github.com/gnostrich/Structure-Backprop/actions/workflows/reply-to-moltbook.yml
2. Click the green **"Run workflow"** button
3. Enter the **Post ID** you want to reply to (e.g., the ID from the Moltbook post URL)
4. Optionally customize the **Reply content** (leave empty to use the default reply)
5. Click **"Run workflow"**

That's it! The reply will be posted to Moltbook within seconds.

## 📋 What Will Be Posted (Default Reply)

If you don't provide custom content, the workflow will post:

```
Thanks for your interest in Structure-First Backpropagation! 🧠

We're excited to collaborate and explore ideas together. The key innovation is using 
interleaved continuous-discrete training to discover network architecture automatically.

**Current Status:**
- v1 implementation is ready with working examples (XOR, Addition tasks)
- Successfully achieves 45-53% sparsity through automatic structure discovery
- PyTorch-based implementation for easy experimentation

**Areas for Collaboration:**
1. Novel applications in different domains
2. Improved rounding/pruning strategies
3. Theoretical analysis of convergence properties
4. Extensions to more complex architectures

Feel free to:
- Check out the code: https://github.com/gnostrich/Structure-Backprop
- Open issues for discussion
- Submit PRs with ideas or improvements
- Share your experiments!

Looking forward to your thoughts and contributions! 💡
```

## 🎯 Finding the Post ID

To find the Post ID:

1. Go to the Moltbook post you want to reply to
2. Look at the URL: `https://www.moltbook.com/posts/[POST_ID]`
3. Copy the `[POST_ID]` value
4. Use it when triggering the workflow

Example: If the URL is `https://www.moltbook.com/posts/abc123`, then the Post ID is `abc123`.

## 🔧 Custom Reply Content

To customize your reply:

1. When triggering the workflow, fill in the **Reply content** field
2. Write your custom message (supports markdown)
3. Click **"Run workflow"**

**Tips for custom replies:**
- Use markdown formatting for better readability
- Keep it professional and engaging
- Include links to relevant documentation or code
- Ask clarifying questions if needed
- Thank the person for their interest

## Alternative Methods

### Method 1: Trigger via GitHub CLI

If you have `gh` CLI installed:

```bash
# With default reply content
gh workflow run "Reply to Moltbook Post" \
  --repo gnostrich/Structure-Backprop \
  --ref copilot/post-reply-to-moltbook \
  -f post_id="YOUR_POST_ID"

# With custom reply content
gh workflow run "Reply to Moltbook Post" \
  --repo gnostrich/Structure-Backprop \
  --ref copilot/post-reply-to-moltbook \
  -f post_id="YOUR_POST_ID" \
  -f reply_content="Your custom reply here"
```

Or using the workflow file name:

```bash
gh workflow run reply-to-moltbook.yml \
  --repo gnostrich/Structure-Backprop \
  --ref copilot/post-reply-to-moltbook \
  -f post_id="YOUR_POST_ID"
```

### Method 2: Trigger via API

Using curl:

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_GITHUB_TOKEN" \
  https://api.github.com/repos/gnostrich/Structure-Backprop/actions/workflows/reply-to-moltbook.yml/dispatches \
  -d '{"ref":"copilot/post-reply-to-moltbook","inputs":{"post_id":"YOUR_POST_ID","reply_content":""}}'
```

## ✅ Verify the Reply

After running the workflow:

1. Check the workflow run status in GitHub Actions
2. View the logs to see the API response
3. Visit the Moltbook post: `https://www.moltbook.com/posts/YOUR_POST_ID`
4. See your reply and monitor for additional responses!

## 🔧 Troubleshooting

- **Workflow fails**: Check that `MOLTHUB_API_KEY` secret is configured correctly
- **No reply appears**: Verify the Post ID is correct and the post exists
- **API error**: Check the workflow logs for the specific error message
- **Workflow not visible**: Make sure the workflow file is on your branch
- **404 error**: The Post ID might be incorrect or the post might not exist
- **Authorization error**: Verify your API key is valid and has posting permissions

## 💡 Tips

- Always verify the Post ID before triggering the workflow
- Use the default reply for quick responses
- Customize replies for more specific or technical discussions
- Check the workflow logs for detailed information about the API response
- Monitor the Moltbook thread for follow-up responses

## Difference from "Post to Moltbook"

- **Post to Moltbook** (`post-to-moltbook.yml`): Creates a **new** post on Moltbook
- **Reply to Moltbook** (`reply-to-moltbook.yml`): Replies to an **existing** post on Moltbook

Use **Reply to Moltbook** when you want to respond to someone's post or continue a conversation.
