module.exports = {
  apps: [
    {
      name: "gopro-s3-automation",
      script: "./s3_automation.py",
      interpreter: "python3",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      env: {
        // Set to "1" if using a public bucket, or "0" to use AWS IAM roles/creds
        PUBLIC_BUCKET: "1",
        // Prevents Python from buffering stdout so you can see pm2 logs instantly
        PYTHONUNBUFFERED: "1"
      }
    }
  ]
};
