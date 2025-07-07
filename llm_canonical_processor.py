import pandas as pd
import numpy as np
import os
import json
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMCanonicalProcessor:
    def __init__(self):
        """Initialize the LLM processor with OpenAI configuration."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '100'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '5'))
        self.batch_delay = float(os.getenv('BATCH_DELAY', '1.0'))
        
        # Initialize LangChain client
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=self.api_key
        )
        
        logger.info(f"Initialized LLM processor with model: {self.model}")
    
    def load_clustering_results(self, file_path):
        """Load the clustering results from CSV."""
        logger.info(f"Loading clustering results from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Filter out singletons (Cluster ID = -1)
        clustered_df = df[df['Cluster ID'] >= 0].copy()
        singletons_df = df[df['Cluster ID'] == -1].copy()
        
        logger.info(f"Total vendors: {len(df)}")
        logger.info(f"Clustered vendors: {len(clustered_df)}")
        logger.info(f"Singleton vendors: {len(singletons_df)}")
        
        return df, clustered_df, singletons_df
    
    def generate_cluster_prompts(self, clustered_df):
        """Generate LLM prompts for each cluster."""
        logger.info("Generating LLM prompts for clusters...")
        
        # Group by cluster ID
        cluster_groups = clustered_df.groupby('Cluster ID')
        
        cluster_prompts = []
        
        for cluster_id, group in cluster_groups:
            vendor_names = group['Vendor Name'].tolist()
            cleaned_names = group['Cleaned Name'].tolist()
            
            # Create prompt
            prompt = f"""Vendor Cluster:
{chr(10).join(vendor_names)}

What is the cleanest, most canonical name for reporting purposes? Respond with only the canonical name, no additional text."""
            
            cluster_info = {
                'cluster_id': int(cluster_id),
                'vendor_count': len(vendor_names),
                'vendor_names': vendor_names,
                'cleaned_names': cleaned_names,
                'prompt': prompt
            }
            
            cluster_prompts.append(cluster_info)
        
        logger.info(f"Generated {len(cluster_prompts)} cluster prompts")
        return cluster_prompts
    
    def process_llm_prompts(self, cluster_prompts):
        """Process all LLM prompts and get canonical names."""
        logger.info(f"Processing {len(cluster_prompts)} LLM prompts...")
        
        results = []
        total_cost = 0.0
        
        # Process in batches
        for i in range(0, len(cluster_prompts), self.batch_size):
            batch = cluster_prompts[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(cluster_prompts) + self.batch_size - 1)//self.batch_size}")
            
            for cluster_info in batch:
                try:
                    # Call LLM
                    response = self.llm.invoke([HumanMessage(content=cluster_info['prompt'])])
                    canonical_name = response.content.strip()
                    
                    # Estimate cost (rough calculation for GPT-3.5-turbo)
                    input_tokens = len(cluster_info['prompt'].split()) * 1.3  # Rough estimate
                    output_tokens = len(canonical_name.split()) * 1.3
                    cost = (input_tokens * 0.0000015) + (output_tokens * 0.000002)  # GPT-3.5-turbo pricing
                    
                    result = {
                        'cluster_id': cluster_info['cluster_id'],
                        'vendor_count': cluster_info['vendor_count'],
                        'vendor_names': cluster_info['vendor_names'],
                        'canonical_name': canonical_name,
                        'llm_response': response.content,
                        'processing_cost': cost
                    }
                    
                    results.append(result)
                    total_cost += cost
                    
                    logger.info(f"Cluster {cluster_info['cluster_id']}: {canonical_name} (${cost:.6f})")
                    
                except Exception as e:
                    logger.error(f"Error processing cluster {cluster_info['cluster_id']}: {e}")
                    # Fallback to first vendor name
                    result = {
                        'cluster_id': cluster_info['cluster_id'],
                        'vendor_count': cluster_info['vendor_count'],
                        'vendor_names': cluster_info['vendor_names'],
                        'canonical_name': cluster_info['vendor_names'][0],
                        'llm_response': f"Error: {str(e)}",
                        'processing_cost': 0.0
                    }
                    results.append(result)
            
            # Delay between batches
            if i + self.batch_size < len(cluster_prompts):
                time.sleep(self.batch_delay)
        
        logger.info(f"Completed LLM processing. Total cost: ${total_cost:.6f}")
        return results, total_cost
    
    def save_results(self, results, total_cost, output_files=True):
        """Save the processing results."""
        # Save LLM results
        llm_results_file = "comprehensive_vendor_results_llm.json"
        with open(llm_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"LLM results saved to: {llm_results_file}")
        
        # Save prompts to file
        if output_files:
            prompts_file = "vendor_cluster_prompts.txt"
            with open(prompts_file, 'w') as f:
                f.write("VENDOR CLUSTER PROMPTS FOR LLM PROCESSING\n")
                f.write("=" * 50 + "\n\n")
                
                for result in results:
                    f.write(f"CLUSTER {result['cluster_id']} ({result['vendor_count']} vendors)\n")
                    f.write("-" * 30 + "\n")
                    f.write("Vendor Cluster:\n\n")
                    for vendor_name in result['vendor_names']:
                        f.write(f"{vendor_name}\n")
                    f.write("\nWhat is the cleanest, most canonical name for reporting purposes?\n")
                    f.write("\n" + "=" * 50 + "\n\n")
            
            logger.info(f"Cluster prompts saved to: {prompts_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("LLM PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total clusters processed: {len(results)}")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Average cost per cluster: ${total_cost/len(results):.6f}")
        
        # Show first 5 results
        print(f"\nFirst 5 canonical names:")
        print("-" * 40)
        for i, result in enumerate(results[:5]):
            print(f"Cluster {result['cluster_id']}: {result['canonical_name']}")
            print(f"  Vendors: {', '.join(result['vendor_names'])}")
            print()
    
    def add_canonical_names_to_dataframe(self, df, results):
        """Add canonical names back to the original dataframe."""
        logger.info("Adding canonical names to dataframe...")
        
        # Create mapping from cluster_id to canonical_name
        canonical_mapping = {result['cluster_id']: result['canonical_name'] for result in results}
        
        # Add canonical name column
        df['Canonical Name'] = df['Cluster ID'].map(canonical_mapping)
        
        # For singletons, use original vendor name as canonical name
        df.loc[df['Cluster ID'] == -1, 'Canonical Name'] = df.loc[df['Cluster ID'] == -1, 'Vendor Name']
        
        # Save updated dataframe
        output_file = "comprehensive_vendor_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Updated dataframe saved to: {output_file}")
        
        return df

def main():
    """Main execution function."""
    try:
        # Initialize processor
        processor = LLMCanonicalProcessor()
        
        # Load clustering results
        df, clustered_df, singletons_df = processor.load_clustering_results("processed_vendors_with_clustering.csv")
        
        # Generate prompts
        cluster_prompts = processor.generate_cluster_prompts(clustered_df)
        
        # Process with LLM
        results, total_cost = processor.process_llm_prompts(cluster_prompts)
        
        # Save results
        processor.save_results(results, total_cost)
        
        # Add canonical names to dataframe
        updated_df = processor.add_canonical_names_to_dataframe(df, results)
        
        print(f"\n✅ Processing complete! Check the output files for results.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 