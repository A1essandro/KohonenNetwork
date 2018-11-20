using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork
{
    internal static class LayerGenerator
    {

        public static InputLayer GenerateInputLayer(int qty, bool withBias)
        {
            var result = new InputLayer();
            for (var i = 0; i < qty; i++)
            {
                result.AddNode(new InputNode());
            }

            if (withBias)
            {
                result.AddNode(new Bias());
            }

            return result;
        }

        public static Layer GenerateOutputLayer(int qty)
        {
            var result = new Layer();
            for (var i = 0; i < qty; i++)
            {
                result.AddNode(new Neuron());
            }

            return result;
        }

    }
}