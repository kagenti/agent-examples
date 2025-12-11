# Cloud Storage APIs Tool

This is an MCP server for accessing cloud storage APIs. It provides a unified interface to interact with various cloud storage providers such as AWS S3, Google Cloud Storage, and Azure Blob Storage.

## Tools
The server has 2 main tools:
- `get_objects`: Lists all objects in a specified bucket/container.
- `perform_action`: Performs action (copy or move) between two cloud storage locations.

## Environment Variables

Configure the server with the following environment variables based on which cloud provider(s) you need to access.

### Google Cloud Storage (GCS)

1. Create a service account in Google Cloud Console
2. Grant it appropriate permissions (e.g., `Storage Object Viewer`, `Storage Object Admin`)
3. Download the JSON key file to your local machine
4. Set the environment variable to the **file path**:

```bash
export GCP_SERVICE_ACCOUNT_KEY="/path/to/your/service-account-key.json"
export GCP_PROJECT_ID="your-gcp-project-id"
```

### AWS S3

1. Create an IAM user or role with appropriate S3 permissions
2. Generate access keys in the AWS IAM console
3. Set the following environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1
```

### Azure Blob Storage

1. Get your storage account connection string from Azure Portal
2. Set the environment variable:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=...;EndpointSuffix=core.windows.net"
```

## Startup

Run the server with:
```bash
uv run cloud_storage_tool.py
```

With MCP inspector:
```bash
uv run fastmcp dev cloud_storage_tool.py
```

## Usage Examples

### Listing Objects

```python
# Google Cloud Storage
get_objects("gs://my-gcs-bucket")

# AWS S3
get_objects("s3://my-s3-bucket")

# Azure Blob Storage
get_objects("azure://my-container")
```

### Copying Files

```python
# Copy within the same provider
perform_action(
    file_uri="gs://source-bucket/path/to/file.txt",
    action="copy",
    target_uri="gs://target-bucket/destination-folder/"
)
```

### Moving Files

```python
# Move within the same provider
perform_action(
    file_uri="s3://source-bucket/file.pdf",
    action="move",
    target_uri="s3://target-bucket/new-location/"
)
```

## Required Permissions

### Google Cloud Storage
- `storage.objects.list` - To list objects
- `storage.objects.get` - To read/download objects
- `storage.objects.create` - To copy objects
- `storage.objects.delete` - To move objects (delete source after copy)

### AWS S3
- `s3:ListBucket` - To list objects
- `s3:GetObject` - To read objects
- `s3:PutObject` - To copy objects
- `s3:DeleteObject` - To move objects (delete source after copy)

### Azure Blob Storage
- `Storage Blob Data Reader` - To list and read blobs
- `Storage Blob Data Contributor` - To copy, move, and delete blobs