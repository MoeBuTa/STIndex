# Deployment Options

## Option 1: Direct Vercel Deployment (Recommended ⭐)

**Pros:**
- ✅ Simplest setup - no GitHub Actions needed
- ✅ Automatic deployments on every push
- ✅ Built-in preview deployments for PRs
- ✅ Easy to manage environment variables in Vercel UI
- ✅ Free for personal projects

**Setup:**
1. Connect GitHub repo to Vercel at https://vercel.com/new
2. Set environment variable in Vercel dashboard:
   - Project Settings → Environment Variables
   - Add `NEXT_PUBLIC_MAPBOX_TOKEN` = your token
   - Select all environments
3. Done! Vercel auto-deploys on every push to main

**When to use:** Most cases - this is the standard Next.js deployment workflow

---

## Option 2: GitHub Actions → Vercel (Advanced)

**Pros:**
- ✅ Full control over deployment process
- ✅ Can run tests, linting, etc. before deployment
- ✅ Can deploy multiple services in one workflow
- ✅ Environment variables managed in GitHub Secrets

**Cons:**
- ❌ More complex setup
- ❌ Need to manage Vercel API token
- ❌ Duplicate work (GitHub Actions + Vercel both do similar things)

**Setup:**

### 1. Get Vercel Token
```bash
# Install Vercel CLI
npm install -g vercel

# Login and generate token
vercel login
# Go to https://vercel.com/account/tokens
# Create a new token
```

### 2. Link your project to Vercel
```bash
cd frontend
vercel link
# Select your project
# This creates .vercel folder with project.json
```

### 3. Add GitHub Secrets
Go to GitHub repo → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:
- `VERCEL_TOKEN` = your Vercel token from step 1
- `VERCEL_ORG_ID` = from `frontend/.vercel/project.json`
- `VERCEL_PROJECT_ID` = from `frontend/.vercel/project.json`
- `NEXT_PUBLIC_MAPBOX_TOKEN` = your Mapbox token

### 4. Use the workflow
The workflow file `.github/workflows/deploy-vercel.yml` is already created.

**When to use:**
- You need custom build steps
- You're deploying multiple services
- You want all deployment logic in one place (GitHub Actions)

---

## Option 3: Hybrid (Best of Both)

Use Vercel for automatic deployments but keep `NEXT_PUBLIC_MAPBOX_TOKEN` in GitHub Secrets:

**Setup:**
1. Connect GitHub repo to Vercel
2. Add `NEXT_PUBLIC_MAPBOX_TOKEN` to GitHub Secrets
3. In Vercel project settings, disable automatic deployments
4. Use simplified GitHub Actions workflow that only passes the token to Vercel

**Note:** This doesn't really save anything - if you're using Vercel, just set the variable in Vercel directly.

---

## Recommendation

**For your use case:**
- If you just want the site deployed: **Use Option 1** (Direct Vercel)
- If you need CI/CD with tests/linting: **Use Option 2** (GitHub Actions)

**Security Note:**
Both approaches are equally secure:
- GitHub Secrets: Encrypted, only accessible during workflow runs
- Vercel Environment Variables: Encrypted, only accessible during builds
- `NEXT_PUBLIC_*` variables are embedded in the client bundle (not truly secret)

For truly secret keys (API keys, database passwords), use server-side environment variables without the `NEXT_PUBLIC_` prefix and access them only in server components or API routes.
