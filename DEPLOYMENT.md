# Production Deployment Guide

This guide provides step-by-step instructions for deploying the ADR Prediction System to production.

## Architecture Overview

- **Frontend**: Next.js 16 → Deployed to Vercel
- **Backend**: FastAPI with Tesseract OCR → Deployed to Render (Docker container) OR Vercel Containers (paid)
- **Why Render for Backend?**: The backend requires Tesseract OCR (a system-level package) for PDF text extraction. Vercel's serverless functions cannot install system packages. Options:
  - **Render (Free)**: Docker web service with full system package support
  - **Vercel Containers (Paid)**: Requires Vercel Pro plan ($20/month) for container deployment

## Prerequisites

- GitHub account with repository: https://github.com/andrewsundaradhas/adr
- Vercel account (free tier works)
- Render account (free tier works)
- Git installed locally

---

## Part 1: Deploy Backend (Choose Option A or B)

### Option A: Deploy to Render (Free - Recommended)

Follow these steps for free deployment with full system package support.

### Step 1: Push Code to GitHub (Already Done)

Your code is already pushed to: https://github.com/andrewsundaradhas/adr

### Step 2: Create Render Account

1. Go to https://render.com
2. Sign up/login with GitHub
3. Verify your email

### Step 3: Deploy Backend via Render Blueprint

1. In Render dashboard, click **"New +"** → **"Blueprint"**
2. Connect your GitHub repository: `andrewsundaradhas/adr`
3. Render will automatically detect `adr_prediction/backend/render.yaml`
4. Review the configuration:
   - **Service Type**: Web Service
   - **Runtime**: Docker
   - **Root Directory**: `adr_prediction/backend`
   - **Dockerfile**: `./Dockerfile`
   - **Health Check Path**: `/health`
5. Click **"Apply"** to deploy

### Step 4: Configure Environment Variables (After Deployment)

1. Once deployed, go to your service dashboard
2. Click **"Environment"** tab
3. Add/verify these environment variables:
   ```
   ADR_ALLOWED_ORIGINS=*  # (Will update after frontend deployment)
   ADR_MAX_UPLOAD_MB=15
   ```
4. Click **"Save Changes"**

### Step 5: Note Your Backend URL

After deployment completes, note your backend URL from the Render dashboard:
- Example: `https://adr-prediction-api.onrender.com`
- Test health endpoint: `https://your-backend-url.onrender.com/health`

### Option B: Deploy to Vercel Containers (Paid - $20/month Pro Plan)

If you prefer to use Vercel for both frontend and backend, you'll need Vercel Pro plan for container deployment.

#### Prerequisites
- Vercel Pro plan ($20/month)
- Docker installed locally

#### Steps

1. **Upgrade to Vercel Pro**:
   - Go to Vercel dashboard → Settings → Billing
   - Upgrade to Pro plan

2. **Create Docker Configuration**:
   - The existing `Dockerfile` in `adr_prediction/backend/` is already configured
   - No additional changes needed

3. **Deploy Container**:
   ```bash
   cd adr_prediction/backend
   vercel --prod
   ```
   - Select "Container" when prompted
   - Vercel will build and deploy the Docker container

4. **Add Environment Variables**:
   - In Vercel project settings → Environment Variables
   - Add: `ADR_ALLOWED_ORIGINS=*` (update after frontend deployment)
   - Add: `ADR_MAX_UPLOAD_MB=15`

5. **Note Your Backend URL**:
   - Example: `https://your-backend.vercel.app`

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Create Vercel Account

1. Go to https://vercel.com
2. Sign up/login with GitHub
3. Verify your email

### Step 2: Import Project to Vercel

1. In Vercel dashboard, click **"Add New..."** → **"Project"**
2. Import your GitHub repository: `andrewsundaradhas/adr`
3. Configure the project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `adr_prediction/frontend`
   - **Build Command**: `npm install && npm run build`
   - **Output Directory**: `.next`
4. Click **"Create"**

### Step 3: Add Environment Variable

1. In the project settings, go to **"Environment Variables"**
2. Add:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com
   ```
   (Replace with your actual backend URL from Part 1 - either Render or Vercel Containers)
3. Click **"Save"**

### Step 4: Redeploy

1. Go to **"Deployments"** tab
2. Click the **"..."** menu on the latest deployment
3. Select **"Redeploy"**
4. This ensures the environment variable is picked up

### Step 5: Note Your Frontend URL

After deployment completes, note your frontend URL:
- Example: `https://adr-prediction.vercel.app`

---

## Part 3: Update CORS Configuration

### Step 1: Update Backend CORS

**If using Render:**
1. Go back to your Render backend service
2. Go to **"Environment"** tab
3. Update `ADR_ALLOWED_ORIGINS`:
   ```
   ADR_ALLOWED_ORIGINS=https://your-frontend-url.vercel.app
   ```
   (Replace with your actual Vercel frontend URL)
4. Click **"Save Changes"**
5. Render will automatically redeploy with the new setting

**If using Vercel Containers:**
1. Go to your Vercel backend project
2. Go to **Settings** → **Environment Variables**
3. Update `ADR_ALLOWED_ORIGINS`:
   ```
   ADR_ALLOWED_ORIGINS=https://your-frontend-url.vercel.app
   ```
4. Click **"Save"**
5. Redeploy the backend project

---

## Part 4: Verify Deployment

### Test Backend Health

```bash
curl https://your-backend-url.onrender.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "2026-07-20T..."
}
```

### Test Frontend

1. Open your Vercel frontend URL in browser
2. Upload a sample prescription PDF
3. Verify the dashboard displays the risk report

### Test API Integration

```bash
curl -X POST "https://your-backend-url.onrender.com/analyze-prescription" \
  -F "file=@/path/to/prescription.pdf"
```

---

## Part 5: Generate Sample PDFs for Testing

If you need sample prescription PDFs for testing:

```bash
cd adr_prediction/backend
python scripts/generate_sample_pdf.py
```

This creates:
- `tests/fixtures/low_risk.pdf`
- `tests/fixtures/high_risk_interaction.pdf`

---

## Troubleshooting

### Backend Issues

**Issue**: Health endpoint returns `{"status":"degraded","model_loaded":false}`
- **Solution**: The Docker build should train the model automatically. If not, check Render build logs for training errors.

**Issue**: OCR not working
- **Solution**: The Dockerfile installs Tesseract OCR. Verify the build completed successfully.

**Issue**: CORS errors in browser
- **Solution**: Ensure `ADR_ALLOWED_ORIGINS` includes your Vercel frontend URL exactly (no trailing slash).

### Frontend Issues

**Issue**: API calls failing with network errors
- **Solution**: Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel environment variables.

**Issue**: Build fails
- **Solution**: Check Vercel build logs. Ensure all dependencies are in `package.json`.

### General Issues

**Issue**: Slow cold starts on Render
- **Solution**: Render free tier has cold starts. This is normal. Consider upgrading to paid tier for production.

---

## Cost Summary

### Vercel (Frontend)
- **Free Tier**: Included
  - 100GB bandwidth/month
  - Unlimited deployments
  - Serverless functions

### Render (Backend)
- **Free Tier**: Included
  - 750 hours/month
  - 0.1 CPU
  - 512MB RAM
  - Sleeps after 15min inactivity (cold starts)

**Note**: For production use, consider upgrading Render to paid tier ($7/month) to avoid cold starts and sleep mode.

---

## Monitoring

### Backend Health

Monitor your Render service:
- Go to Render dashboard → Your service → **"Metrics"**
- Monitor CPU, memory, and response times

### Frontend Analytics

Monitor your Vercel deployment:
- Go to Vercel dashboard → Your project → **"Analytics"**
- Monitor page views, bandwidth, and build times

---

## Security Notes

1. **CORS**: After deployment, update `ADR_ALLOWED_ORIGINS` to your specific Vercel domain instead of `*`
2. **API Rate Limiting**: Consider adding rate limiting to the backend for production
3. **File Upload Limits**: Current limit is 15MB (configurable via `ADR_MAX_UPLOAD_MB`)
4. **HTTPS**: Both Vercel and Render provide automatic HTTPS

---

## Next Steps After Deployment

1. **Test thoroughly** with various prescription PDFs
2. **Monitor performance** during initial usage
3. **Set up alerts** in Render for service health
4. **Consider custom domain** for both frontend and backend
5. **Review logs** regularly for errors or unusual patterns

---

## Support

For issues:
- Backend: Check Render logs in dashboard
- Frontend: Check Vercel build logs in dashboard
- GitHub Issues: https://github.com/andrewsundaradhas/adr/issues
