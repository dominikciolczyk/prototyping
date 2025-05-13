import boto3
import json
import sys
import time
from zenml.client import Client
from botocore.exceptions import ClientError

# ── CONFIG ───────────────────────────────────────────────
project = "cloud-resource-prediction"
bucket_name = f"{project}-artifacts"
region = "eu-central-1"
user_name = f"{project}-user"
policy_name = "CloudResourcePredictionPolicy"
artifact_store_name = "s3_artifact_store"
# ────────────────────────────────────────────────────────

# Initialize AWS clients
iam = boto3.client("iam")
s3 = boto3.client("s3", region_name=region)

# 1️⃣ Create or ensure S3 bucket exists
def ensure_bucket(name, region):
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=name)
        else:
            s3.create_bucket(
                Bucket=name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print(f"✅ Bucket '{name}' created.")
    except ClientError as e:
        if e.response['Error']['Code'] in ['BucketAlreadyOwnedByYou', 'BucketAlreadyExists']:
            print(f"ℹ️ Bucket '{name}' already exists.")
        else:
            print(f"❌ Failed to create bucket: {e}")
            sys.exit(1)

# 2️⃣ Create or ensure IAM user exists
def ensure_user(name):
    try:
        iam.create_user(UserName=name)
        print(f"✅ IAM user '{name}' created.")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"ℹ️ IAM user '{name}' already exists.")

# 3️⃣ Create or ensure policy exists and attach to user
def ensure_policy_and_attach(user, policy_name, bucket):
    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": [
                f"arn:aws:s3:::{bucket}",
                f"arn:aws:s3:::{bucket}/*"
            ]
        }]
    }
    # Find existing policy
    arn = None
    for p in iam.list_policies(Scope='Local')['Policies']:
        if p['PolicyName'] == policy_name:
            arn = p['Arn']
            break
    # Create if missing
    if not arn:
        arn = iam.create_policy(
            PolicyName=policy_name,
            PolicyDocument=json.dumps(policy_doc)
        )['Policy']['Arn']
        print(f"✅ Policy '{policy_name}' created.")
    else:
        print(f"ℹ️ Policy '{policy_name}' already exists.")
    # Attach
    attached = iam.list_attached_user_policies(UserName=user)['AttachedPolicies']
    if not any(p['PolicyArn'] == arn for p in attached):
        iam.attach_user_policy(UserName=user, PolicyArn=arn)
        print(f"✅ Policy '{policy_name}' attached to user '{user}'.")
    else:
        print(f"ℹ️ Policy '{policy_name}' already attached to user.")

# 4️⃣ Create new access key for user if none exists
def ensure_access_keys(user):
    keys = iam.list_access_keys(UserName=user)['AccessKeyMetadata']
    if keys:
        ak = keys[0]['AccessKeyId']
        print(f"ℹ️ Access key '{ak}' already exists for user '{user}'.")
        # We cannot retrieve secret; assume CLI is configured already
        return
    cred = iam.create_access_key(UserName=user)['AccessKey']
    print("🔑 New AWS credentials:")
    print(f"  AWS_ACCESS_KEY_ID={cred['AccessKeyId']}")
    print(f"  AWS_SECRET_ACCESS_KEY={cred['SecretAccessKey']}")

# 5️⃣ Setup ZenML artifact store and stack
def configure_zenml(bucket, store_name):
    client = Client()
    # Wait for dashboard if necessary
    timeout = 5
    for _ in range(timeout):
        try:
            client.list_stacks()
            break
        except Exception:
            print("⌛ Waiting for ZenML server...")
            time.sleep(1)
    # Install s3 integration
    try:
        client.integration_install("s3")
    except Exception:
        pass

if __name__ == "__main__":
    ensure_bucket(bucket_name, region)
    ensure_user(user_name)
    ensure_policy_and_attach(user_name, policy_name, bucket_name)
    ensure_access_keys(user_name)
    configure_zenml(bucket_name, artifact_store_name)
    print("🎉 Setup complete. Your project is ready.")
