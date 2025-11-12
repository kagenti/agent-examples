import json
import logging
import os
import sys
from typing import List, Dict, Any, Tuple
from fastmcp import FastMCP
from google.cloud import storage
from google.oauth2 import service_account
import boto3
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format='%(levelname)s: %(message)s')


# GCP credentials
GCP_SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# AWS credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Azure credentials
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

def parse_cloud_uri(uri: str) -> Tuple[str, str, str]:
    """Parse cloud storage URI and return (provider, bucket/container, path)."""
    if uri.startswith("gs://"):
        parts = uri.replace("gs://", "").split("/", 1)
        return "gcs", parts[0], parts[1] if len(parts) > 1 else ""
    elif uri.startswith("s3://"):
        parts = uri.replace("s3://", "").split("/", 1)
        return "s3", parts[0], parts[1] if len(parts) > 1 else ""
    elif uri.startswith("azure://"):
        parts = uri.replace("azure://", "").split("/", 1)
        return "azure", parts[0], parts[1] if len(parts) > 1 else ""
    else:
        # If no scheme, raise error
        raise ValueError(f"Invalid cloud storage URI: {uri}")

def get_gcs_client():
    """Create and return a GCS client using service account credentials."""
    try:
        if GCP_SERVICE_ACCOUNT_KEY is None:
            logger.error("GCP_SERVICE_ACCOUNT_KEY environment variable not set")
            return None
        
        # Parse service account key from JSON string or file path
        if GCP_SERVICE_ACCOUNT_KEY.startswith("{"):
            # It's a JSON string
            credentials_info = json.loads(GCP_SERVICE_ACCOUNT_KEY)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
        else:
            # It's a file path
            credentials = service_account.Credentials.from_service_account_file(GCP_SERVICE_ACCOUNT_KEY)
        
        client = storage.Client(credentials=credentials, project=GCP_PROJECT_ID)
        logger.info("Successfully authenticated with GCP")
        return client
    except Exception as e:
        logger.error(f"Error authenticating with GCP: {e}")
        return None

def get_s3_client():
    """Create and return an S3 client using AWS credentials."""
    try:
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
        else:
            # Use default credentials (IAM role, environment, etc.)
            client = boto3.client('s3', region_name=AWS_REGION)
        
        logger.info("Successfully authenticated with AWS S3")
        return client
    except Exception as e:
        logger.error(f"Error authenticating with AWS S3: {e}")
        return None

def get_azure_blob_service_client():
    """Create and return an Azure Blob Service client."""
    try:
        if AZURE_STORAGE_CONNECTION_STRING:
            client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        elif AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
            account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
            client = BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_ACCOUNT_KEY)
        else:
            logger.error("Azure credentials not configured")
            return None
        
        logger.info("Successfully authenticated with Azure Blob Storage")
        return client
    except Exception as e:
        logger.error(f"Error authenticating with Azure Blob Storage: {e}")
        return None

def list_objects_unified(provider: str, bucket_or_container: str) -> List[Dict[str, Any]]:
    """List objects from any cloud provider."""
    objects = []
    
    if provider == "gcs":
        storage_client = get_gcs_client()
        if not storage_client:
            raise Exception("Could not authenticate with GCP")
        
        bucket = storage_client.bucket(bucket_or_container)
        blobs = bucket.list_blobs()
        
        for blob in blobs:
            objects.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "storage_class": blob.storage_class,
                "public_url": blob.public_url
            })
    
    elif provider == "s3":
        s3_client = get_s3_client()
        if not s3_client:
            raise Exception("Could not authenticate with AWS S3")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_or_container):
            for obj in page.get('Contents', []):
                objects.append({
                    "name": obj['Key'],
                    "size": obj['Size'],
                    "content_type": None,
                    "created": obj['LastModified'].isoformat() if 'LastModified' in obj else None,
                    "updated": obj['LastModified'].isoformat() if 'LastModified' in obj else None,
                    "storage_class": obj.get('StorageClass'),
                    "public_url": f"s3://{bucket_or_container}/{obj['Key']}"
                })
    
    elif provider == "azure":
        azure_client = get_azure_blob_service_client()
        if not azure_client:
            raise Exception("Could not authenticate with Azure Blob Storage")
        
        container_client = azure_client.get_container_client(bucket_or_container)
        blobs = container_client.list_blobs()
        
        for blob in blobs:
            objects.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_settings.content_type if blob.content_settings else None,
                "created": blob.creation_time.isoformat() if blob.creation_time else None,
                "updated": blob.last_modified.isoformat() if blob.last_modified else None,
                "storage_class": blob.blob_tier,
                "public_url": f"azure://{bucket_or_container}/{blob.name}"
            })
    
    return objects

def copy_object_unified(provider: str, source_bucket: str, source_path: str, 
                       target_bucket: str, target_path: str) -> bool:
    """Copy object within the same cloud provider."""
    if provider == "gcs":
        storage_client = get_gcs_client()
        if not storage_client:
            raise Exception("Could not authenticate with GCP")
        
        source_bucket_obj = storage_client.bucket(source_bucket)
        source_blob = source_bucket_obj.blob(source_path)
        
        if not source_blob.exists():
            raise Exception(f"Source file does not exist: gs://{source_bucket}/{source_path}")
        
        target_bucket_obj = storage_client.bucket(target_bucket)
        source_bucket_obj.copy_blob(source_blob, target_bucket_obj, target_path)
        return True
    
    elif provider == "s3":
        s3_client = get_s3_client()
        if not s3_client:
            raise Exception("Could not authenticate with AWS S3")
        
        copy_source = {'Bucket': source_bucket, 'Key': source_path}
        s3_client.copy_object(CopySource=copy_source, Bucket=target_bucket, Key=target_path)
        return True
    
    elif provider == "azure":
        azure_client = get_azure_blob_service_client()
        if not azure_client:
            raise Exception("Could not authenticate with Azure Blob Storage")
        
        source_blob_client = azure_client.get_blob_client(container=source_bucket, blob=source_path)
        target_blob_client = azure_client.get_blob_client(container=target_bucket, blob=target_path)
        
        if not source_blob_client.exists():
            raise Exception(f"Source file does not exist: azure://{source_bucket}/{source_path}")
        
        target_blob_client.start_copy_from_url(source_blob_client.url)
        return True
    
    return False

def delete_object_unified(provider: str, bucket_or_container: str, path: str) -> bool:
    """Delete object from any cloud provider."""
    if provider == "gcs":
        storage_client = get_gcs_client()
        if not storage_client:
            raise Exception("Could not authenticate with GCP")
        
        bucket = storage_client.bucket(bucket_or_container)
        blob = bucket.blob(path)
        blob.delete()
        return True
    
    elif provider == "s3":
        s3_client = get_s3_client()
        if not s3_client:
            raise Exception("Could not authenticate with AWS S3")
        
        s3_client.delete_object(Bucket=bucket_or_container, Key=path)
        return True
    
    elif provider == "azure":
        azure_client = get_azure_blob_service_client()
        if not azure_client:
            raise Exception("Could not authenticate with Azure Blob Storage")
        
        blob_client = azure_client.get_blob_client(container=bucket_or_container, blob=path)
        blob_client.delete_blob()
        return True
    
    return False

def download_text_unified(provider: str, bucket_or_container: str, path: str) -> str:
    """Download text content from any cloud provider."""
    if provider == "gcs":
        storage_client = get_gcs_client()
        if not storage_client:
            raise Exception("Could not authenticate with GCP")
        
        bucket = storage_client.bucket(bucket_or_container)
        blob = bucket.blob(path)
        
        if not blob.exists():
            raise Exception(f"File does not exist: gs://{bucket_or_container}/{path}")
        
        return blob.download_as_text()
    
    elif provider == "s3":
        s3_client = get_s3_client()
        if not s3_client:
            raise Exception("Could not authenticate with AWS S3")
        
        response = s3_client.get_object(Bucket=bucket_or_container, Key=path)
        return response['Body'].read().decode('utf-8')
    
    elif provider == "azure":
        azure_client = get_azure_blob_service_client()
        if not azure_client:
            raise Exception("Could not authenticate with Azure Blob Storage")
        
        blob_client = azure_client.get_blob_client(container=bucket_or_container, blob=path)
        
        if not blob_client.exists():
            raise Exception(f"File does not exist: azure://{bucket_or_container}/{path}")
        
        return blob_client.download_blob().readall().decode('utf-8')
    
    raise Exception(f"Unsupported provider: {provider}")

# Create FastMCP app
mcp = FastMCP("CloudStorage")

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_objects(bucket_uri: str) -> str:
    """Get all objects from a cloud storage bucket/container."""
    try:
        # Parse URI to determine provider and bucket
        provider, bucket_name, _ = parse_cloud_uri(bucket_uri)
        
        logger.debug(f"Getting objects from {provider} bucket '{bucket_name}'")
        
        # Get the raw list of objects
        objects = list_objects_unified(provider, bucket_name)
        
        # Loop through and enrich each object with the full file_uri
        for obj in objects:
            # This assumes your list_objects_unified returns dicts
            # and that the object key is stored in the 'name' field.
            # Adjust 'name' if your key is stored differently (e.g., 'key')
            if 'name' in obj:
                obj['file_uri'] = f"{provider}://{bucket_name}/{obj['name']}"
            else:
                logger.warning(f"Object {obj} missing 'name' key, cannot construct file_uri")

        logger.debug(f"Successfully retrieved and processed {len(objects)} objects from {provider} bucket '{bucket_name}'")
        
        return json.dumps({
            "provider": provider,
            "bucket": bucket_name,
            "object_count": len(objects),
            "objects": objects
        })
    
    except Exception as e:
        logger.error(f"Error listing objects: {e}")
        return json.dumps({"error": f"Failed to list objects: {str(e)}"})

@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False})
def perform_action(file_uri: str, action: str, target_uri: str) -> str:
    """
    Performs the configured action (move or copy) between cloud storage locations.
    
    Args:
        file_uri: Source file URI (e.g., 'gs://bucket/path/file.txt')
        action: Action to perform - either 'move' or 'copy'
        target_uri: Target folder URI (e.g., 'gs://bucket/folder/'). Must end with '/' for folder.
    """
    logger.debug(f"Performing action '{action}' from '{file_uri}' to '{target_uri}'")

    # Validate action
    if action not in ["move", "copy"]:
        return json.dumps({"error": f"Invalid action '{action}'. Must be 'move' or 'copy'"})
    
    # Validate target is a folder (ends with /)
    if not target_uri.endswith("/"):
        return json.dumps({"error": f"Target URI must be a folder path ending with '/': {target_uri}"})
    
    try:
        # Parse source and target URIs
        source_provider, source_bucket, source_path = parse_cloud_uri(file_uri)
        target_provider, target_bucket, target_folder = parse_cloud_uri(target_uri)
        
        # Ensure providers match
        if source_provider != target_provider:
            return json.dumps({"error": f"Cross-provider operations not supported. Source is {source_provider}, target is {target_provider}"})
        
        # Extract filename from source path
        filename = os.path.basename(source_path)
        
        # Construct full target blob path (folder + filename)
        target_path = os.path.join(target_folder, filename).replace("\\", "/")
        
        # Construct full URIs for response
        full_source_uri = f"{source_provider}://{source_bucket}/{source_path}"
        full_target_uri = f"{target_provider}://{target_bucket}/{target_path}"
        
        # Perform copy operation
        copy_object_unified(source_provider, source_bucket, source_path, target_bucket, target_path)
        
        result = {
            "action": action,
            "provider": source_provider,
            "source": full_source_uri,
            "target": full_target_uri,
            "target_folder": f"{target_provider}://{target_bucket}/{target_folder}",
            "filename": filename,
            "status": "success"
        }
        
        # If action is move, delete the source
        if action == "move":
            delete_object_unified(source_provider, source_bucket, source_path)
            logger.debug(f"Successfully moved '{full_source_uri}' to '{full_target_uri}'")
            result["message"] = f"File moved from {full_source_uri} to {full_target_uri}"
        else:
            logger.debug(f"Successfully copied '{full_source_uri}' to '{full_target_uri}'")
            result["message"] = f"File copied from {full_source_uri} to {full_target_uri}"
        
        return json.dumps(result)
    
    except Exception as e:
        logger.error(f"Error performing {action} operation: {e}")
        return json.dumps({"error": f"Failed to {action} file: {str(e)}"})

def run_server():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    configured_providers = []
    if GCP_SERVICE_ACCOUNT_KEY and GCP_PROJECT_ID:
        configured_providers.append("GCP")
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        configured_providers.append("AWS S3")
    if AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY):
        configured_providers.append("Azure")
    
    if not configured_providers:
        logger.warning("No cloud provider credentials configured. Please set up at least one provider.")
    else:
        logger.info(f"Configured providers: {', '.join(configured_providers)}")
    
    run_server()