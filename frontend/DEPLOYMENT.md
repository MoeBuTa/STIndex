# Frontend Deployment Guide

> **Quick Start:** For most users, deploy directly to Vercel and set environment variables in the Vercel dashboard. See [DEPLOYMENT_OPTIONS.md](./DEPLOYMENT_OPTIONS.md) for alternative approaches.

## Environment Variables Setup

### Local Development

1. Copy the environment template:
   ```bash
   cd frontend
   cp .env.example .env.local
   ```

2. Get your Mapbox access token:
   - Visit https://account.mapbox.com/access-tokens/
   - Create a new token or copy an existing one

3. Add the token to `.env.local`:
   ```
   NEXT_PUBLIC_MAPBOX_TOKEN=pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJ5b3VyLXRva2VuIn0...
   ```

4. Restart the development server:
   ```bash
   npm run dev
   ```

### Production Deployment

> **Important:** GitHub repository secrets are for **GitHub Actions only**. They are NOT available at runtime in your Next.js app. You must set environment variables in your deployment platform (Vercel, Netlify, etc.).

#### Vercel (Recommended)

1. Push your code to GitHub
2. Import the project in Vercel: https://vercel.com/new
3. Add environment variables in Vercel dashboard:
   - Go to Project Settings → Environment Variables
   - Add `NEXT_PUBLIC_MAPBOX_TOKEN` with your token value
   - Select all environments (Production, Preview, Development)
4. Deploy

#### Netlify

1. Push your code to GitHub
2. Import the project in Netlify: https://app.netlify.com/start
3. Add environment variables in Netlify dashboard:
   - Go to Site settings → Build & deploy → Environment
   - Add `NEXT_PUBLIC_MAPBOX_TOKEN` with your token value
4. Deploy

#### Docker / Self-hosted

When building the Docker image, pass the environment variable:

```bash
docker build --build-arg NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here -t stindex-frontend .
```

Or use a `.env.production.local` file (not committed to git):

```bash
# .env.production.local
NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here
```

Then build and run:

```bash
npm run build
npm run start
```

## Important Notes

- **Never commit** `.env.local` or `.env.production.local` files to git
- GitHub repository secrets are for **GitHub Actions** only, not for runtime environment variables
- Next.js requires `NEXT_PUBLIC_` prefix for client-side environment variables
- Environment variables are embedded at **build time**, so rebuild after changing them

## Troubleshooting

### Map not loading?

1. Check browser console for errors
2. Verify the token is set: `console.log(process.env.NEXT_PUBLIC_MAPBOX_TOKEN)`
3. Ensure you rebuilt after adding the token: `npm run build`
4. Check Mapbox token is valid and has the correct permissions

### Token not available in production?

1. Verify the environment variable is set in your deployment platform
2. Check the variable name matches exactly: `NEXT_PUBLIC_MAPBOX_TOKEN`
3. Rebuild and redeploy after adding the variable
