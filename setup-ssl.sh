#!/bin/bash
# SSL Setup Script for Crack Inspection AI
# Domain: crackinspection.duckdns.org

set -e

DOMAIN="crackinspection.duckdns.org"
EMAIL="nasriabidi55@gmail.com"  # Change this to your email

echo "=== SSL Setup for $DOMAIN ==="

# Step 1: Stop existing containers (to free port 80)
echo "[1/6] Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Step 2: Install certbot
echo "[2/6] Installing Certbot..."
apt-get update
apt-get install -y certbot

# Step 3: Create webroot directory
echo "[3/6] Creating certbot webroot..."
mkdir -p /var/www/certbot

# Step 4: Get SSL certificate using standalone mode
echo "[4/6] Obtaining SSL certificate..."
certbot certonly \
    --standalone \
    --preferred-challenges http \
    --agree-tos \
    --no-eff-email \
    --email $EMAIL \
    -d $DOMAIN

# Step 5: Verify certificate
echo "[5/6] Verifying certificate..."
if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "✓ SSL certificate obtained successfully!"
    ls -la /etc/letsencrypt/live/$DOMAIN/
else
    echo "✗ Certificate not found. Please check the certbot output above."
    exit 1
fi

# Step 6: Start containers
echo "[6/6] Starting containers with SSL..."
docker-compose up --build -d

echo ""
echo "=== Setup Complete ==="
echo "Your app is now available at: https://$DOMAIN"
echo ""
echo "To auto-renew certificates, add this cron job:"
echo "0 0 1 * * certbot renew --quiet && docker-compose restart web"
