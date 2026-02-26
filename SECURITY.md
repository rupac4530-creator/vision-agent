# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ Yes    |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Email**: Send details to **rupac4530@gmail.com**
2. **Do NOT** open a public issue for security vulnerabilities
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgement**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix/Patch**: Prioritized based on severity

## Security Best Practices

This project follows these security practices:

- ❌ No API keys or secrets are committed to the repository
- ✅ All sensitive values are managed via `.env` files (not in git)
- ✅ `.gitignore` excludes `.env`, model weights, and uploaded files
- ✅ Feature flags disable external services by default
- ✅ Input validation on all API endpoints
