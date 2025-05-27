# requirements.txt
Flask==2.3.3
Flask-CORS==4.0.0

# ===================================
# config.json (example - create this file)
{
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "from_email": "your-email@gmail.com",
  "from_password": "your-app-password",
  "from_name": "Vision 2054",
  "welcome_email_enabled": true,
  "admin_notifications": true,
  "admin_email": "admin@vision2054.org"
}

# ===================================
# docker-compose.yml (optional - for easy deployment)
version: '3.8'
services:
  vision2054-backend:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

# ===================================
# Dockerfile (optional - for containerization)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data

EXPOSE 5000

CMD ["python", "app.py"]

# ===================================
# run.sh (helper script)
#!/bin/bash
echo "🎵 Starting Vision 2054 Backend..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create data directory
mkdir -p data

# Check if config exists
if [ ! -f "data/config.json" ]; then
    echo "⚠️  Creating default config file..."
    echo "Please edit data/config.json with your email settings!"
fi

# Run the app
python app.py

# ===================================
# Frontend Integration JavaScript
# Add this to your HTML file to connect to the backend

const API_BASE_URL = 'http://localhost:5000/api';

// Replace your existing email signup handler with this:
document.getElementById('email-signup-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = e.target.querySelector('input[type="email"]').value;
    const submitButton = e.target.querySelector('button');
    const successMessage = document.getElementById('success-message');
    
    // Update button state
    submitButton.disabled = true;
    submitButton.textContent = 'Signing up...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/signup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: email })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Success!
            successMessage.innerHTML = `
                ✨ ${data.message}<br>
                <small>You're subscriber #${data.total_signups}!</small>
            `;
            successMessage.style.display = 'block';
            e.target.reset();
            
            // Optional: Track the signup
            console.log('Signup successful:', data);
            
        } else {
            // Handle errors
            if (response.status === 409) {
                successMessage.innerHTML = '📧 You\'re already signed up! Thank you for your enthusiasm.';
                successMessage.style.display = 'block';
                e.target.reset();
            } else {
                throw new Error(data.error || 'Signup failed');
            }
        }
        
    } catch (error) {
        console.error('Signup error:', error);
        alert(`Error: ${error.message}\n\nPlease try again or contact support.`);
    } finally {
        // Reset button
        submitButton.disabled = false;
        submitButton.textContent = 'Sign Me Up';
        
        // Hide success message after 10 seconds
        setTimeout(() => {
            successMessage.style.display = 'none';
        }, 10000);
    }
});

// Optional: Add signup stats display
async function updateSignupStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        if (response.ok) {
            // You can display this somewhere on your page
            console.log(`Total signups: ${data.total_signups}`);
            console.log(`Recent signups: ${data.recent_signups}`);
        }
    } catch (error) {
        console.log('Stats unavailable:', error);
    }
}

// Call on page load
updateSignupStats();

# ===================================
# production.py (production-ready version)
import os
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )

# ===================================
# README.md
# Vision 2054 Email Backend

Simple, self-hosted email signup backend for Vision 2054.

## Features

- ✅ Email validation and deduplication
- ✅ CSV storage (no database required)
- ✅ Automatic welcome emails
- ✅ Admin notifications
- ✅ CORS enabled for frontend
- ✅ Export functionality
- ✅ Simple configuration

## Quick Start

1. **Install Python 3.7+**

2. **Setup Backend:**
   ```bash
   # Clone/create your project directory
   mkdir vision2054-backend
   cd vision2054-backend
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install requirements
   pip install flask flask-cors
   
   # Copy app.py to your directory
   # Run the app
   python app.py
   ```

3. **Configure Email:**
   - Edit `data/config.json` with your email settings
   - For Gmail: Enable 2FA and create App Password
   - Use the App Password, not your regular password

4. **Update Frontend:**
   - Add the JavaScript code to your HTML
   - Change `API_BASE_URL` to your backend URL

## Email Setup (Gmail)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Create App Password:**
   - Go to Google Account Settings
   - Security → 2-Step Verification → App passwords
   - Select "Mail" and generate password
3. **Update config.json:**
   ```json
   {
     "from_email": "your-email@gmail.com",
     "from_password": "your-16-char-app-password"
   }
   ```

## Deployment Options

### Option 1: Simple VPS (DigitalOcean, Linode, etc.)
```bash
# On your server
git clone your-repo
cd vision2054-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Option 2: Heroku
```bash
# Create Procfile
echo "web: python production.py" > Procfile

# Deploy
heroku create vision2054-backend
git push heroku main
```

### Option 3: Railway/Render
- Connect your GitHub repo
- Set start command: `python app.py`
- Add environment variables if needed

## API Endpoints

- `POST /api/signup` - Submit email signup
- `GET /api/stats` - Get signup statistics  
- `GET /api/export` - Download CSV (admin)
- `GET /health` - Health check

## Data Storage

- **Location**: `data/signups.csv`
- **Format**: CSV with timestamp, email, IP, user agent
- **Backup**: Regular backups recommended

## Security Notes

- Add authentication for admin endpoints in production
- Use HTTPS in production
- Consider rate limiting
- Regular backups of CSV file

## Cost

- **Hosting**: $5-20/month (VPS) or free tier (Heroku/Railway)
- **Email**: Free (using your Gmail) or SMTP service
- **Total**: Under $20/month for thousands of signups
