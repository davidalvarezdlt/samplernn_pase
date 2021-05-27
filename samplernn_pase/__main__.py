import skeltorch
import samplernn_pase.data
import samplernn_pase.runner

# Create and run Skeltorch project
skeltorch.Skeltorch(samplernn_pase.data.SampleRNNPASEData(), samplernn_pase.runner.SampleRNNPASERunner()).run()
