# Crack Inspection AI - Docker Setup

AI-powered crack detection system using YOLOv8 deep learning.

## Project Structure

```
app/
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile.api          # Backend API Dockerfile
├── Dockerfile.web          # Frontend Dockerfile
├── nginx.conf              # Nginx configuration
├── .dockerignore           # Docker ignore file
├── main.py                 # FastAPI backend
├── requirements.txt        # Python dependencies
├── model/
│   └── best.pt             # YOLOv8 trained model
└── webapp/
    ├── index.html          # Frontend HTML
    ├── styles.css          # Frontend styles
    ├── script.js           # Frontend JavaScript
    └── 50x.html            # Error page
```

## Quick Start

### Prerequisites
- Docker & Docker Compose installed
- At least 4GB of RAM available

### Build and Run

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access the Application

- **Web App**: http://localhost:80
- **API Health**: http://localhost:8000/health
- **API Direct**: http://localhost:8000/predict

## Services

| Service | Port | Description |
|---------|------|-------------|
| web | 80 | Nginx serving frontend + API proxy |
| api | 8000 | FastAPI backend with YOLOv8 |

## Architecture

```
User Browser
     │
     ▼
┌─────────────┐
│   Nginx     │ ← Port 80
│  (web)      │
└──────┬──────┘
       │ /api/* requests
       ▼
┌─────────────┐
│  FastAPI    │ ← Port 8000
│   (api)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  YOLOv8     │
│   Model     │
└─────────────┘
```

## AWS EC2 Deployment

1. SSH into your EC2 instance
2. Install Docker and Docker Compose
3. Clone/copy the project files
4. Run `docker-compose up --build -d`
5. Open port 80 in Security Group

### Security Group Rules

| Type | Port | Source |
|------|------|--------|
| HTTP | 80 | 0.0.0.0/0 |
| Custom TCP | 8000 | 0.0.0.0/0 (optional) |

## Useful Commands

```bash
# Rebuild specific service
docker-compose build api
docker-compose build web

# Restart services
docker-compose restart

# View running containers
docker-compose ps

# Check API logs
docker-compose logs api

# Enter container shell
docker-compose exec api bash
docker-compose exec web sh

# Remove all containers and images
docker-compose down --rmi all
```

## Troubleshooting

### API not responding
```bash
# Check if container is running
docker-compose ps

# Check API logs
docker-compose logs api
```

### Model not loading
- Ensure `model/best.pt` exists
- Check if model file is valid

### Out of memory
- Increase Docker memory limit
- Reduce batch size in model

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| PYTHONUNBUFFERED | 1 | Python output buffering |

## License

MIT License
