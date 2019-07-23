using NeuralNetwork.Structure.Common;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Networks;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Synapses;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen
{
    public abstract class TwoLayersNetwork<TLayer> : ISimpleNetwork
    {

        #region Private fields

        private IReadOnlyLayer<IMasterNode> _inputLayer;
        private IReadOnlyLayer<INotInputNode> _outputLayer;
        private readonly ICollection<ISynapse> _synapses = new List<ISynapse>();
        private IDictionary<INode, int> _outputPositions;
        private double[] _output;
        private SemaphoreSlim _processingLocker = new SemaphoreSlim(1, 1);

        #endregion;

        #region Events

        public event Action<IEnumerable<double>> OnOutput;

        public event Func<IInput<IEnumerable<double>>, IEnumerable<double>, Task> OnInput;

        public event Func<ISimpleNetwork, IEnumerable<double>, Task> OnResultCalculated;

        #endregion

        #region Public properties

        public virtual IReadOnlyLayer<IMasterNode> InputLayer
        {
            get => _inputLayer;
            set
            {
                _inputLayer = value;
                _inputLayer.InsertInto(this);
            }
        }

        public virtual IReadOnlyLayer<INotInputNode> OutputLayer
        {
            get => _outputLayer;
            set
            {
                _outputLayer = value;
                _outputLayer.InsertInto(this);

                _outputPositions = new Dictionary<INode, int>();
                _output = new double[_outputLayer.Nodes.Count()];
                var i = 0;
                foreach (var node in _outputLayer.Nodes)
                {
                    _outputPositions.Add(node, i++);
                    node.OnResultCalculated += _processOutput;
                }
            }
        }

        public virtual ICollection<ISynapse> Synapses => _synapses;

        /// <summary>
        /// All layers from input to output
        /// </summary>
        public IEnumerable<IReadOnlyLayer<INode>> Layers
        {
            get
            {
                yield return InputLayer;
                yield return OutputLayer;
            }
        }

        public virtual IEnumerable<double> LastCalculatedValue { get; protected set; }

        #endregion

        #region ctors

        public TwoLayersNetwork(IReadOnlyLayer<IMasterNode> inputLayer, IReadOnlyLayer<INotInputNode> outputLayer)
        {
            InputLayer = inputLayer;
            OutputLayer = outputLayer;
        }

        #endregion

        /// <summary> 
        /// Write input value to each input-neuron (<see cref="IInput{double}"/>) in input-layer.
        /// </summary>
        /// <param name="input"></param>
        public virtual async Task Input(IEnumerable<double> input)
        {
            Contract.Requires(input != null, nameof(input));

            try
            {
                await _processingLocker.WaitAsync();
                _output = new double[_outputLayer.Nodes.Count()];

                await _processInput(input);

                LastCalculatedValue = _output;
                if (OnResultCalculated != null)
                    await OnResultCalculated(this, _output);
            }
            finally
            {
                _processingLocker.Release();
            }
        }

        public virtual async Task<IEnumerable<double>> Output()
        {
            try
            {
                await _processingLocker.WaitAsync();

                OnOutput?.Invoke(_output);

                return _output;
            }
            finally
            {
                _processingLocker.Release();
            }
        }

        public void AddSynapse(ISynapse synapse)
        {
            Contract.Requires(synapse != null, nameof(synapse));

            _synapses.Add(synapse);

            synapse.InsertInto(this);
        }

        #region private methods

        private async Task _processInput(IEnumerable<double> input)
        {
            if (OnInput != null)
                await OnInput(this, input);

            await _inputToNodes(input);
        }

        private Task _inputToNodes(IEnumerable<double> input)
        {
            var inputNodes = _inputLayer.Nodes.OfType<IInputNode>().ToArray();
            var taskList = new List<Task>(inputNodes.Length);

            var index = 0;
            foreach (var value in input)
            {
                taskList.Add(inputNodes[index++].Input(value));
            }

            return Task.WhenAll(taskList);
        }

        private Task _processOutput(INode node, double value)
        {
            var position = _outputPositions[node];
            _output[position] = value;

            return Task.CompletedTask;
        }

        #endregion

    }

}
