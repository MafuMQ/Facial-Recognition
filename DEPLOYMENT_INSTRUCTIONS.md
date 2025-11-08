# Deploying Flask Facial Recognition App on Google Compute Engine (Ubuntu)

## 1. Set Up Google Cloud Project and SSH

> **Tip for new users:**
> If you encounter permission issues or need to set a password for your user (for `su -` or other root actions), you can set a password with:
> ```sh
> sudo passwd $USER
> ```
> Then switch to the root user (if needed and permitted) with:
> ```sh
> su -
> ```
> If you do not have sudo/root access, use the Google Cloud Console to adjust permissions or contact your administrator.

1. Set your project:
   ```sh
   gcloud config set project YOUR_PROJECT_ID
   ```
2. SSH into your VM (replace `USER_NAME`, `YOUR_INSTANCE_NAME`, and `YOUR_ZONE`):
   ```sh
   gcloud compute ssh USER_NAME@YOUR_INSTANCE_NAME --zone YOUR_ZONE
   ```
   Example:
   ```sh
   gcloud compute ssh USER_NAME@instance-20250101-123456 --zone us-central1-f
   ```

## 2. Update System and Install Dependencies

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev git
```

## 3. Clone Your Repository

```sh
git clone <your-repo-url>
cd Facial-Recognition
```

## 4. Set Up Python Virtual Environment

```sh
python3 -m venv venv
source venv/bin/activate
```

## 5. Install Python Requirements

```sh
pip install --upgrade pip
pip install setuptools
pip install -r requirements.txt
```

> **Note:** The `setuptools` package is required for `face_recognition_models` to work properly with newer Python versions (3.13+).

## 6. Configure Firewall for External Access

**Option 1: Using gcloud command (recommended)**

Allow external traffic on port 5000 (for Flask development server):
```sh
gcloud compute firewall-rules create allow-flask --allow tcp:5000 --source-ranges 0.0.0.0/0
```

Or for production with Gunicorn on port 8000:
```sh
gcloud compute firewall-rules create allow-gunicorn --allow tcp:8000 --source-ranges 0.0.0.0/0
```

**Option 2: Using Google Cloud Console (manual method)**

1. Go to Google Cloud Console → VPC network → Firewall
2. Click "Create Firewall Rule"
3. Set:
   - Name: `allow-flask` (or `allow-gunicorn`)
   - Direction: Ingress
   - Action: Allow
   - Targets: All instances in the network
   - Source IP ranges: `0.0.0.0/0`
   - Protocols and ports: Check "Specified protocols and ports" → TCP → Enter `5000` (or `8000`)
4. Click "Create"

## 7. Run Your Flask App

```sh
python app.py
```

Your app will be available at:
- **Local**: http://127.0.0.1:5000
- **External**: http://YOUR_EXTERNAL_IP:5000

> **Important:** Use **HTTP** (not HTTPS) to access your app. The Flask development server runs on HTTP by default.

## 8. (Optional) Run with Gunicorn for Production

```sh
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## 9. (Optional) Set Up Nginx as Reverse Proxy

- Install Nginx:
  ```sh
  sudo apt install nginx
  ```
- Configure Nginx to proxy requests to Gunicorn (edit `/etc/nginx/sites-available/default`):
  ```nginx
  server {
      listen 80;
      server_name _;

      location / {
          proxy_pass http://127.0.0.1:8000;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }
  }
  ```
- Restart Nginx:
  ```sh
  sudo systemctl restart nginx
  ```

---

**Note:**
- If you encounter permission issues, check your user’s privileges or use the Google Cloud Console to adjust settings.
- For `face_recognition` and similar dependencies, system packages like CMake and build-essential are required.
- Always specify your username when using `gcloud compute ssh` for best results.
