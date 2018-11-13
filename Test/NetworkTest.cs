using System;
using Xunit;
using KohonenNetwork;
using System.Linq;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Layers;

namespace Test
{
    public class NetworkTest
    {

        [Fact]
        public void LayersTest()
        {
            var network = new KohonenNetwork<Gaussian>(5, 2);

            Assert.Equal(1, network.Layers.Count());
            Assert.Throws<NotSupportedException>(() => network.Layers.Add(new Layer()));
            Assert.Equal(1, network.Layers.Count());
            Assert.True(network.Layers != network.Layers);
        }

        [Fact]
        public void NodesTest()
        {
            var network0 = new KohonenNetwork<Gaussian>(5, 2, false);
            var network1 = new KohonenNetwork<Gaussian>(5, 2, true);

            Assert.Equal(5, network0.InputLayer.Nodes.Count());
            Assert.Equal(6, network1.InputLayer.Nodes.Count());
            Assert.Equal(2, network0.OutputLayer.Nodes.Count());
        }
    }
}