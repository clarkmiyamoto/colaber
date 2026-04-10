# colaber
Instead of running python scripts on your local, have it run on Google Colab. GPU support included.

Running python goes from
```bash
python main.py --args
```
To this
```bash
colaber main.py --args
```

## Setup

### Installation
1. Git clone this
2. Run `pip install . -e`

### OAuth2 Credentials

Colaber authenticates with Google Colab using OAuth2 credentials from Google's
Colab VS Code extension. You need to provide these as environment variables.

1. Copy the example env file:
   ```bash
   cp .env.example .env
   ```

2. Obtain the client ID and secret from the
   [Colab VS Code extension source](https://github.com/googlecolab/colab-vscode).
   They are the same public OAuth2 credentials embedded in Google's official
   extension — search the extension source for `client_id` and `client_secret`.

3. Fill in your `.env` file:
   ```
   COLAB_CLIENT_ID=<your-client-id>
   COLAB_CLIENT_SECRET=<your-client-secret>
   ```

> **Note:** The `.env` file is git-ignored and will not be committed.
