using System;

namespace KohonenNetwork
{
    public class NetworkConfiguration
    {

        public int InputLayerNodes { get; set; }

        public int OutputLayerNodes { get; set; }

        public bool CreateBiasNode { get; set; } //false by default

        public NetworkConfiguration(int inputNodes, int outputNodes, bool createBias = false)
        {
            InputLayerNodes = inputNodes;
            OutputLayerNodes = outputNodes;
            CreateBiasNode = createBias;
        }

    }
}