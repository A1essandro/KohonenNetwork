using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork
{
    public class LayerProjetion2D<TNode> where TNode : INode
    {

        public TNode[,] Net { get; }
        private int _w, _h;

        public LayerProjetion2D(ILayer<TNode> layer, int w, int h)
        {
            if (layer.Nodes.Count() != w * h)
                throw new System.ArgumentException("w * h must be equals quantity of layer nodes");

            _w = w;
            _h = h;

            var arr = new TNode[w, h];
            var index = 0;
            for (var x = 0; x < w; x++)
            {
                for (var y = 0; y < h; y++)
                {
                    arr[x, y] = layer.Nodes[index++];
                }
            }

            Net = arr;
        }

        public async Task<double[,]> Output()
        {
            var arr = new double[_w, _h];

            for (var x = 0; x < _w; x++)
            {
                for (var y = 0; y < _h; y++)
                {
                    arr[x, y] = await Net[x, y].Output();
                }
            }

            return arr;
        }

    }
}