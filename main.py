import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

# 1. Load and verify dataset
print("⏳ Loading dataset...")
df = pd.read_csv("products.csv")
print("✅ Dataset loaded successfully")
print("\n📊 Sample data:")
print(df.head(3))
print("\n📋 Available columns:", df.columns)

# 2. Initialize model
print("\n🔧 Initializing embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model ready")

# 3. Pinecone setup
print("\n🌲 Connecting to Pinecone...")
pc = Pinecone(api_key="pcsk_4srGy4_JfHGNmRjzhczwn9ar9aQQXkgxBXnzYmR258tv4aoWv23Ajs2F5C6ZVhMKegqsv9")
index_name = "product-recommend"

# 4. Index management
if index_name not in pc.list_indexes().names():
    print("\n🆕 Creating new index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("⏳ Waiting for index initialization...")
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(2)
else:
    print("\n♻️ Resetting existing index...")
    index = pc.Index(index_name)
    index.delete(delete_all=True)
    time.sleep(5)  # Wait for deletion to complete

index = pc.Index(index_name)

# 5. Data upload with verification
print("\n⬆️ Uploading vectors...")
success_count = 0
for i, row in df.iterrows():
    try:
        vector = model.encode(row["description"]).tolist()
        metadata = {
            "description": str(row["description"]),
            "category": str(row["category"]),
            "unitprice": float(row["unitprice"]),
            "country": str(row["country"])
        }
        response = index.upsert([(str(row["productID"]), vector, metadata)])
        success_count += 1
    except Exception as e:
        print(f"❌ Failed to upload product {row['productID']}: {str(e)}")

print(f"✅ Uploaded {success_count}/{len(df)} vectors successfully")

# 6. Wait for index stabilization
print("\n⏳ Waiting 30 seconds for index stabilization...")
time.sleep(30)

# 7. Enhanced recommendation function
def recommend(query, top_k=3):
    try:
        print(f"\n🔍 Searching for: '{query}'")
        
        # Verify index status
        stats = index.describe_index_stats()
        print(f"📊 Index stats - Vectors: {stats.total_vector_count}")
        
        if stats.total_vector_count == 0:
            print("❌ Error: Index is empty!")
            return
        
        # Generate and verify query vector
        query_vector = model.encode(query).tolist()
        print("🔢 Query vector generated")
        
        # Execute query
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        print(f"🔎 Found {len(results.matches)} matches")
        
        if not results.matches:
            print("❌ No matches found!")
            return
        
        # Display results
        print(f"\n🏆 Top {top_k} recommendations:")
        for i, match in enumerate(results.matches):
            print(f"\n🎯 Match #{i+1} (Score: {match.score:.4f})")
            print(f"🆔 ID: {match.id}")
            
            if not hasattr(match, 'metadata') or not match.metadata:
                print("📭 No metadata available")
                continue
                
            print("📋 Metadata:")
            print(f"  📝 Description: {match.metadata.get('description', 'N/A')}")
            print(f"  🏷️ Category: {match.metadata.get('category', 'N/A')}")
            print(f"  💰 Price: ₹{match.metadata.get('unitprice', 'N/A')}")
            print(f"  🌍 Country: {match.metadata.get('country', 'N/A')}")
    
    except Exception as e:
        print(f"🔥 Error: {str(e)}")
        import traceback
        traceback.print_exc()

# 8. Test recommendations
print("\n" + "="*50)
print("🚀 Starting Recommendation Tests")
print("="*50)

test_queries = [
    "bluetooth speaker",
    "fitness gear",
    "eco-friendly products",
    "electronic gadgets",
    "kitchen items"
]

for query in test_queries:
    recommend(query)
    print("\n" + "-"*50 + "\n")

print("✅ All tests completed")