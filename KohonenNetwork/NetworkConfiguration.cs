using System;

namespace KohonenNetwork
{
    public class NetworkConfiguration
    {

        public int InputLayerNodes { get; set; }

        public int OutputLayerNodes { get; set; }

        public bool CreateBiasNode { get; set; } //false by default

        public Func<double> SynapseWeightGenerator { get; set; } = () => rand.NextDouble();

        private static Random rand = new Random();

        public NetworkConfiguration(int inputNodes, int outputNodes, bool createBias = false)
        {
            InputLayerNodes = inputNodes;
            OutputLayerNodes = outputNodes;
            CreateBiasNode = createBias;
        }

    }
}