/**
 * Example Integration: Using Cloud-Hosted GPT-OSS 120B with QRNG
 * 
 * This example shows how to integrate the cloud-hosted model with your existing
 * Gaia Quantum Nexus system for true quantum-influenced text generation.
 */

import { QuantumBlockchainsQRNG } from '../server/services/qrng';
import { CloudModelManager, CloudCostEstimator } from '../server/services/cloud-model-providers';

// Configuration
const CONFIG = {
  // Choose your cloud provider: 'modal', 'runpod', 'together', 'replicate', 'sagemaker'
  CLOUD_PROVIDER: process.env.CLOUD_PROVIDER || 'modal',
  
  // Modal is recommended for cost-effectiveness
  MODAL_ENDPOINT: process.env.MODAL_ENDPOINT || 'https://your-username--gpt-oss-120b-qrng.modal.run',
  MODAL_API_KEY: process.env.MODAL_API_KEY || '',
  
  // RunPod for dedicated low-latency
  RUNPOD_ENDPOINT: process.env.RUNPOD_ENDPOINT || '',
  RUNPOD_API_KEY: process.env.RUNPOD_API_KEY || '',
  
  // QRNG configuration
  QRNG_API_KEY: process.env.QRNG_API_KEY || '',
  
  // Model parameters
  VOCAB_SIZE: 50000, // GPT-OSS vocabulary size
  MAX_TOKENS: 100,
  TEMPERATURE: 0.7
};

/**
 * Production integration for GPT-OSS 120B with QRNG logit modification
 */
async function runProductionIntegration() {
  console.log('=== GPT-OSS 120B Production Integration with QRNG ===\n');
  console.log('Model: bartowski/openai_gpt-oss-120b-GGUF-MXFP4-Experimental\n');
  
  // Step 1: Show cost estimates
  console.log('📊 Cost Estimates for Different Providers:\n');
  ['modal', 'runpod', 'together', 'replicate', 'sagemaker'].forEach(provider => {
    const estimate = CloudCostEstimator.estimate(provider, 30); // 30 hours/month
    console.log(`${estimate.provider}:`);
    console.log(`  Monthly Cost: ${estimate.monthlyCost} (30 hours usage)`);
    console.log(`  Specs: ${estimate.specifications}`);
    console.log(`  Latency: ${estimate.latency}\n`);
  });
  
  // Step 2: Initialize QRNG
  console.log('🔮 Initializing Quantum Random Number Generator...');
  const qrng = new QuantumBlockchainsQRNG();
  const isQRNGAvailable = await qrng.isAvailable();
  
  if (!isQRNGAvailable) {
    console.error('❌ QRNG not available. Please set QRNG_API_KEY environment variable.');
    console.log('Get your API key at: https://qrng.qbck.io');
    return;
  }
  console.log('✅ QRNG initialized and available\n');
  
  // Step 3: Initialize Cloud Model Manager
  console.log(`☁️  Initializing Cloud Model Manager with ${CONFIG.CLOUD_PROVIDER} provider...`);
  const modelManager = new CloudModelManager(qrng, CONFIG.CLOUD_PROVIDER);
  console.log('✅ Cloud Model Manager initialized\n');
  
  // Step 4: Demonstrate quantum-influenced generation
  const prompt = "Explain how quantum consciousness emerges from";
  console.log(`📝 Prompt: "${prompt}"\n`);
  
  try {
    console.log('⚡ Generating with QRNG-modified inference...');
    const startTime = Date.now();
    
    // Generate text with quantum influence
    const result = await modelManager.generateWithQuantumInfluence(
      prompt,
      CONFIG.VOCAB_SIZE
    );
    
    const elapsedTime = (Date.now() - startTime) / 1000;
    
    console.log('\n✨ Generation Complete!');
    console.log(`📖 Generated Text: ${result.text}`);
    console.log(`🎲 QRNG Vector Sample: [${result.qrngVector.slice(0, 5).map(v => v.toFixed(3)).join(', ')}...]`);
    console.log(`☁️  Provider Used: ${result.provider}`);
    console.log(`⏱️  Generation Time: ${elapsedTime.toFixed(2)} seconds`);
    console.log(`💰 Estimated Cost: $${(elapsedTime * 0.00265).toFixed(4)}`);
    
  } catch (error) {
    console.error('❌ Generation failed:', error);
    console.log('\n💡 Troubleshooting:');
    console.log('1. Ensure your cloud model is deployed (see deployment/DEPLOYMENT_GUIDE.md)');
    console.log('2. Set the correct environment variables for your provider');
    console.log('3. Verify QRNG API key is valid and has credits');
  }
  
  // Step 5: Show streaming example
  console.log('\n\n=== Streaming Generation Example ===\n');
  console.log('🌊 Starting token stream with QRNG modification...\n');
  
  try {
    let tokenCount = 0;
    const streamStart = Date.now();
    
    for await (const { token, qrngModifier } of modelManager.streamWithQuantumInfluence(prompt, 10)) {
      tokenCount++;
      console.log(`Token ${tokenCount}: "${token}" (QRNG modifier: ${qrngModifier.toFixed(3)})`);
    }
    
    const streamTime = (Date.now() - streamStart) / 1000;
    console.log(`\n✅ Streamed ${tokenCount} tokens in ${streamTime.toFixed(2)} seconds`);
    console.log(`📊 Throughput: ${(tokenCount / streamTime).toFixed(1)} tokens/second`);
    
  } catch (error) {
    console.error('❌ Streaming failed:', error);
  }
}

/**
 * Production low-latency optimization setup
 */
async function setupLowLatencyProduction() {
  console.log('\n\n=== Production Low-Latency QRNG Integration ===\n');
  
  const qrng = new QuantumBlockchainsQRNG();
  
  // Pre-fetch QRNG data for low latency
  console.log('🚀 Pre-fetching QRNG data for low-latency generation...');
  const batchSize = 10000;
  const qrngBuffer = await qrng.getRandomFloats(batchSize, -2, 2);
  console.log(`✅ Pre-fetched ${batchSize} QRNG values\n`);
  
  // Demonstrate buffered generation
  const modelManager = new CloudModelManager(qrng, 'modal');
  
  console.log('⚡ Using buffered QRNG for instant access:');
  const measurements = [];
  
  for (let i = 0; i < 5; i++) {
    const start = Date.now();
    
    // Use pre-fetched QRNG data (no API call needed)
    const vocabSize = 1000;
    const offset = i * vocabSize;
    const qrngSlice = qrngBuffer.slice(offset, offset + vocabSize);
    
    const latency = Date.now() - start;
    measurements.push(latency);
    console.log(`  Request ${i + 1}: QRNG access in ${latency}ms`);
  }
  
  const avgLatency = measurements.reduce((a, b) => a + b, 0) / measurements.length;
  console.log(`\n📊 Average QRNG access latency: ${avgLatency.toFixed(1)}ms`);
  console.log('💡 Tip: Pre-fetching eliminates QRNG API latency from the critical path\n');
}

/**
 * Production deployment checklist
 */
function showDeploymentChecklist() {
  console.log('\n\n=== Production Deployment Checklist ===\n');
  
  const checklist = [
    '☐ Choose cloud provider based on requirements:',
    '   - Modal: Best for cost-effectiveness, serverless scaling',
    '   - RunPod: Best for dedicated low-latency, consistent performance',
    '   - SageMaker: Best for enterprise, SLA requirements',
    '',
    '☐ Deploy the model:',
    '   - Modal: modal deploy deployment/modal-gpt-oss-120b.py',
    '   - RunPod: Upload model to pod, run server',
    '   - See deployment/DEPLOYMENT_GUIDE.md for details',
    '',
    '☐ Configure environment variables:',
    '   - CLOUD_PROVIDER=modal',
    '   - MODAL_ENDPOINT=https://your-endpoint.modal.run',
    '   - MODAL_API_KEY=your-api-key',
    '   - QRNG_API_KEY=your-qrng-key',
    '',
    '☐ Optimize for low latency:',
    '   - Deploy in same region as QRNG API (US-East)',
    '   - Use pre-fetching for QRNG data',
    '   - Keep model containers warm',
    '   - Consider dedicated instances for <100ms latency',
    '',
    '☐ Monitor and scale:',
    '   - Track: latency, throughput, costs, QRNG usage',
    '   - Set up alerts for high latency or errors',
    '   - Implement request queuing for burst traffic',
    '   - Use auto-scaling for variable load',
    '',
    '☐ Security:',
    '   - Rotate API keys regularly',
    '   - Use HTTPS for all connections',
    '   - Implement rate limiting',
    '   - Log all requests for auditing'
  ];
  
  checklist.forEach(item => console.log(item));
}

// Run production integration
async function main() {
  try {
    await runProductionIntegration();
    await setupLowLatencyProduction();
    showDeploymentChecklist();
    
    console.log('\n\n✅ Production integration complete!');
    console.log('📚 Next steps:');
    console.log('1. Review deployment/DEPLOYMENT_GUIDE.md');
    console.log('2. Choose your cloud provider');
    console.log('3. Deploy the model using provided scripts');
    console.log('4. Update environment variables');
    console.log('5. Test the production integration\n');
    
  } catch (error) {
    console.error('Error in production integration:', error);
  }
}

// Export for use in other modules
export {
  CloudModelManager,
  CloudCostEstimator,
  runProductionIntegration,
  setupLowLatencyProduction
};

// Run if executed directly
if (require.main === module) {
  main();
}