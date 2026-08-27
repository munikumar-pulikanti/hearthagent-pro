# Security Policy

## Reporting Security Issues

If you discover a security vulnerability, please email security@example.com instead of using the issue tracker.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide an update within 7 days.

## Supported Versions

| Version | Supported |
|---------|----------|
| 0.1.x   | ✅ |
| < 0.1   | ❌ |

## Security Best Practices

### Local Mode
- Keep `config.yaml` out of version control (use `.gitignore`)
- Don't commit API keys or credentials
- Use strong passwords for MinIO (`minioadmin` is for dev only)

### Production Mode
- Rotate AWS keys regularly
- Use IAM roles instead of access keys when possible
- Enable encryption at rest for S3, DynamoDB, etc.
- Audit cloud backend access logs
- Use VPCs to restrict network access

### Memory Data
- Sensitive data in memories is not encrypted by default
- If handling secrets, use separate encryption layer
- Archive and purge old memories regularly
