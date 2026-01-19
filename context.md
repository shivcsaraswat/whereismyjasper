# Find My Jasper - Project Context

## Overview
A full-stack ML application to identify "Jasper" (a white pet dog) from images using binary classification.

**This is a learning project.** Claude is acting as a **mentor/coach** to help the user gain hands-on experience with Angular, FastAPI, CI/CD, Docker, and cloud deployment. The user will **build each component themselves** with Claude providing guidance, explanations, and code reviews rather than writing everything for them.

## Tech Stack
| Component | Technology |
|-----------|------------|
| ML Service | PyTorch, ResNet50 (transfer learning), FastAPI |
| Backend | FastAPI (Python) |
| Frontend | Angular 17+ |
| Deployment | Google Cloud Run (containerized) |
| CI/CD | GitHub Actions (separate workflow per service) |

## Architecture
Three separate microservices:
1. **ml-service** - Model training and inference API
2. **backend** - API gateway, orchestration
3. **frontend** - Angular UI for image upload and results

## Current Progress

### Completed
- [x] Project directory structure created
- [x] ml-service `pyproject.toml` with entry points (`jasper-train`, `jasper-serve`)
- [x] ml-service `requirements.txt` with `-e .[dev]` for editable install with dev deps
- [x] `ml_service` Python package structure created
- [x] `ml_service/models/classifier.py` - Model creation, save/load, device detection
- [x] `ml_service/training/train.py` - Full training pipeline with data augmentation
- [x] `ml_service/api/main.py` - FastAPI app with /health, /predict endpoints
- [x] `ml-service/Dockerfile` - Multi-stage build, CPU-only PyTorch
- [x] Git repo initialized and pushed to GitHub (shivcsaraswat/whereismyjasper)
- [x] GitHub Actions CI workflow created (`.github/workflows/ml-service-ci.yml`)
- [x] GCP Artifact Registry repository created (`jasper-repo` in `us-central1`)
- [x] GCP Service Account created for GitHub Actions (`github-actions@whereisjasper`)
- [x] GitHub Secrets configured (`GCP_PROJECT_ID`, `GCP_SA_KEY`)
- [x] Docker build step in CI workflow
- [x] Push to GAR step in CI workflow

### In Progress
- [ ] Test full CI pipeline (Docker build + push to GAR)
- [ ] Cloud Run deployment (CD)

### Pending
- [ ] Collect dataset (Jasper images + non-Jasper images)
- [ ] Train model with real data
- [ ] Fix pytest tests (null bytes issue in test files)
- [ ] Backend service
- [ ] Frontend service

## Key Decisions Made
1. **Angular 17+** (not AngularJS 1.x)
2. **Cloud Run** for containerized deployment
3. **Binary classification** (Jasper vs Not Jasper) - may add dog detection later (Phase 2)
4. **Package structure** with `pyproject.toml` and `-e .` for proper Python packaging
5. **Entry points** for CLI commands (`jasper-train`, `jasper-serve`)
6. **Transfer learning strategy**: Feature extraction first (freeze backbone), fine-tune Layer4 if needed
7. **Incremental CI/CD**: Build module → Test → CI/CD → Deploy for each service

## Project Structure
```
Whereisjasper/
├── Context.md              ← You are here
├── .github/workflows/      ← CI/CD (to be created)
├── ml-service/
│   ├── pyproject.toml      ← Package config with entry points
│   ├── requirements.txt    ← Just "-e ."
│   ├── ml_service/         ← Python package
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── main.py     ← FastAPI app (TODO)
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   └── train.py    ← Training script (DONE)
│   │   └── models/
│   │       ├── __init__.py
│   │       └── classifier.py  ← Model utilities (DONE)
│   ├── tests/              ← Test files (to be created)
│   │   └── __init__.py
│   ├── data/
│   │   ├── jasper/         ← PUT JASPER IMAGES HERE
│   │   └── not_jasper/     ← PUT OTHER IMAGES HERE
│   └── models/             ← Trained models saved here
├── backend/
│   └── app/
├── frontend/
└── file_requested/
    └── Resnet_Pretrained_Pytorch.ipynb  ← Original Colab notebook
```

## ML Architecture Notes

### ResNet50 Transfer Learning
```
ResNet50 (pretrained on ImageNet)
├── Stem + Layer1-3: FROZEN (universal features: edges, textures, shapes)
├── Layer4: OPTIONALLY FINE-TUNABLE (high-level features)
└── FC Layer: REPLACED (2048 → 2 for Jasper/Not Jasper)
```

### Why Freeze Backbone?
- Early layers learn universal features (edges, textures) - no need to retrain
- With only ~7 images, training all layers would cause overfitting
- Feature extraction (train only FC layer) is safest for small datasets
- Can unfreeze Layer4 later for fine-tuning if accuracy is insufficient

### Data Augmentation Strategy
With ~7 Jasper images, we use aggressive augmentation:
- RandomResizedCrop (simulates different distances)
- RandomHorizontalFlip (Jasper facing left/right)
- RandomRotation (±15 degrees)
- ColorJitter (brightness, contrast, saturation, hue)

### Future Enhancement: Two-Stage Pipeline
Phase 2 improvement if needed:
1. Object detection (YOLO) to find dogs in image
2. Crop detected dogs → feed to ResNet classifier
This handles: small Jasper in frame, multiple dogs, noisy backgrounds

## Dataset Notes
- Currently have 6-7 real images of Jasper
- Will use data augmentation to expand training set
- Need to collect "not_jasper" images (other dogs, objects, etc.)

## Next Steps (When Resuming)
1. Add images to `data/jasper/` and `data/not_jasper/`
2. Install package: `cd ml-service && pip install -e .`
3. Run training: `jasper-train --epochs 20`
4. Build FastAPI inference endpoint
5. Write unit tests
6. Set up CI/CD with GitHub Actions
7. Dockerize and deploy to Cloud Run

## Useful Commands
```bash
cd ml-service
pip install -e .              # Install package in editable mode
pip install -e ".[dev]"       # Install with dev dependencies (pytest)

jasper-train                  # Train with defaults
jasper-train --epochs 30      # Custom epochs
jasper-train --help           # See all options

jasper-serve                  # Start inference API (TODO)
```

## User Background
- Familiar with Python
- New to Docker (conceptual understanding)
- New to GCP (gcloud CLI installed)
- Learning: FastAPI, Angular, CI/CD, containerization

## Mentorship Approach (UPDATED)
**Important for future Claude instances:** The user's **primary learning goal is CI/CD**.

### Updated Approach (January 2025)
1. **Claude builds** the basic working application (ML service first, then backend/frontend)
2. **User implements** the CI/CD pipeline with Claude's guidance (hands-on learning)
3. **User dives into** the application code afterwards to understand it

This approach lets the user focus on CI/CD with a real, deployable application rather than building everything from scratch first.

### Primary Learning Focus: CI/CD
The user wants hands-on experience with:
- **GitHub Actions** - Workflow syntax, triggers, jobs, environment variables, secrets
- **Docker** - Dockerfile best practices, multi-stage builds, image optimization
- **Google Cloud Run** - Container deployment, service configuration, environment management

### Secondary Learning (After CI/CD)
Once CI/CD is working, dive into:
- **FastAPI** - Request/response models, dependency injection, async endpoints
- **Angular 17+** - Component architecture, services, HTTP client
- **PyTorch** - Transfer learning, model architecture

### Teaching Guidelines for CI/CD
- Claude provides the application code
- **User writes the CI/CD workflows** - Claude guides with explanations
- Explain the **"why"** behind each workflow step
- For each CI/CD component:
  1. Explain the concepts (triggers, jobs, steps, artifacts)
  2. Provide a skeleton/outline
  3. Let the user fill in the details
  4. Review and debug together
- Use "Learning Moment" sections for key CI/CD concepts

### Hands-On Learning Roadmap (Revised)
| Phase | Module | Status | Who Builds | Key Learning Goals |
|-------|--------|--------|------------|-------------------|
| 1 | ML Service - Complete App | DONE | Claude | Working API with placeholder model |
| 2 | ML Service - CI Pipeline | DONE | User | GitHub Actions, path filtering, uv |
| 3 | ML Service - Docker + GAR | DONE | User | Dockerfile, multi-stage builds, Artifact Registry |
| 4 | ML Service - CD Pipeline | IN PROGRESS | User | Cloud Run deployment, secrets |
| 5 | Backend Service | PENDING | Claude | API gateway code |
| 6 | Backend CI/CD | PENDING | User | Multi-service workflows |
| 7 | Frontend Service | PENDING | Claude | Angular app code |
| 8 | Frontend CI/CD | PENDING | User | Static hosting, build optimization |

### Progress Tracking
- DONE = Completed
- IN PROGRESS = Currently working on
- UP NEXT = Ready to start
- PENDING = Not started

**Current Phase:** 4 - Cloud Run deployment (CD pipeline)

## ML Service - Ready for CI/CD ✓

### Files Created by Claude
| File | Purpose |
|------|---------|
| `ml-service/ml_service/api/main.py` | FastAPI app with /health, /predict endpoints |
| `ml-service/tests/test_api.py` | Unit tests (7 tests) |
| `ml-service/tests/conftest.py` | Pytest configuration |
| `ml-service/Dockerfile` | Multi-stage build, CPU-only PyTorch |
| `ml-service/.dockerignore` | Excludes unnecessary files from build |
| `ml-service/.env.example` | Environment variables template |
| `.gitignore` | Project-wide git ignores |

### Environment Variables (for GitHub Secrets)
```
# Application
PORT=8080
ENVIRONMENT=development
LOG_LEVEL=INFO
USE_PLACEHOLDER_MODEL=true

# GCP (required for deployment)
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCP_ARTIFACT_REGISTRY=jasper-registry
GCP_SERVICE_NAME=jasper-ml-service
GCP_SA_KEY=(base64 encoded service account JSON)
```

### Local Testing Commands
```bash
cd ml-service
pip install -e ".[dev]"      # Install with dev dependencies
pytest                        # Run tests
jasper-serve                  # Start local server on port 8080
```

### CI/CD Implementation Progress
1. ~~Initialize git repo and push to GitHub~~ ✓
2. ~~Create GitHub Actions workflow (CI - tests)~~ ✓
3. ~~Add Docker build step~~ ✓
4. ~~Set up GCP Artifact Registry~~ ✓
5. ~~Add push to registry step~~ ✓
6. Set up Cloud Run deployment ← **IN PROGRESS** (first attempt failed - container timeout, Dockerfile fixed)
7. Test full pipeline

## CI/CD Pipeline Architecture

### Target Pipeline (User will implement)
```
Push/PR → GitHub Actions → Tests → Docker Build → Artifact Registry → Cloud Run → Live URL
```

### Pipeline Components
1. **CI (Continuous Integration)**
   - Trigger: Push to main, Pull Requests
   - Steps: Lint, Unit Tests, Integration Tests
   - Tool: GitHub Actions

2. **Build**
   - Dockerize the application
   - Multi-stage build for smaller images
   - CPU-only PyTorch (smaller image size)

3. **Push**
   - Push Docker image to Google Artifact Registry
   - Tag with commit SHA and `latest`

4. **Deploy (CD)**
   - Deploy to Google Cloud Run
   - Auto-scaling, managed infrastructure
   - Result: Public URL for the service

### GCP Resources Needed
- Google Cloud Project (user has this ready)
- Artifact Registry repository
- Cloud Run service
- Service Account with appropriate permissions

### GitHub Secrets Required
- `GCP_PROJECT_ID` - Google Cloud project ID
- `GCP_SA_KEY` - Service Account JSON key (for authentication)
- `GCP_REGION` - Deployment region (e.g., us-central1)

## Current GitHub Actions Workflow
File: `.github/workflows/ml-service-ci.yml`
```yaml
name: CI for ml-service
on:
  push:
    paths:
      - 'ml-service/**'
  pull_request:
    paths:
      - 'ml-service/**'
  workflow_dispatch:

env:
  REGION: us-central1
  REGISTRY: us-central1-docker.pkg.dev
  IMAGE_NAME: jasper-ml-service

jobs:
  ML_Service_CI:
    runs-on: ubuntu-latest
    steps:
      - Checkout code
      - Set Up Python 3.11
      - Install uv + dependencies
      - Placeholder for pytest
      - Authenticate to GCP
      - Setup gcloud
      - Configure Docker for GAR
      - Build and push Docker image to GAR
```

## GCP Configuration
| Resource | Value |
|----------|-------|
| Project ID | `whereisjasper` |
| Region | `us-central1` |
| Artifact Registry | `jasper-repo` |
| Service Account | `github-actions@whereisjasper.iam.gserviceaccount.com` |
| Full Image Path | `us-central1-docker.pkg.dev/whereisjasper/jasper-repo/jasper-ml-service` |

## GitHub Secrets Configured
- `GCP_PROJECT_ID` - whereisjasper
- `GCP_SA_KEY` - Service account JSON key

## Session Notes
- User prefers incremental CI/CD approach: build → test → CI/CD → deploy per module
- User asked excellent questions about ResNet architecture and transfer learning
- Discussed: skip connections, bottleneck blocks, adaptive average pooling, layer freezing
- User understands why we freeze backbone (prevent overfitting with small dataset)
- User proposed future enhancement: object detection + classification pipeline (Phase 2)
- **January 2025**: Changed approach - Claude builds app, user implements CI/CD
- User has Google Cloud account ready for deployment
- **January 2025 (Session 2)**: Implemented CI workflow with Docker build and GAR push
- **January 2025 (Session 3)**: First Cloud Run deployment attempted - failed due to container startup timeout (ResNet50 weights downloading at runtime). Fixed Dockerfile to pre-download weights during build. Deployment still pending.
