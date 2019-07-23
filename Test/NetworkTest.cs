using NeuralNetwork.Kohonen;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using System.Linq;
using Xunit;

namespace Test
{
    public class NetworkTest
    {

        [Fact]
        public void LayersTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 5);
            var outputLayer = new Layer(() => new Neuron(), 2);
            var network = new KohonenNetwork(inputLayer, outputLayer);

            Assert.Equal(2, network.Layers.Count());
        }

        [Fact]
        public void NodesTest()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 5);
            var outputLayer = new Layer(() => new Neuron(), 2);
            var network = new NeuralNetwork.Kohonen.KohonenNetwork(inputLayer, outputLayer);

            Assert.Equal(5, network.InputLayer.Nodes.Count());
            Assert.Equal(2, network.OutputLayer.Nodes.Count());
        }
    }
}