using System.Linq;
using KohonenNetwork;
using NeuralNetworkConstructor.Structure.Nodes;
using Xunit;

namespace Test
{
    public class ProjectionTest
    {

        [Fact]
        public void LayersTest()
        {
            var network = new KohonenNetwork.KohonenNetwork(10, 6);
            var projection = new LayerProjetion2D<INotInputNode>(network.OutputLayer, 3, 2);

            Assert.Equal(network.OutputLayer.Nodes.First(), projection.Net[0, 0]);
        }

    }
}