using Seq2SeqSharp;

namespace RWKV
{
    public class RunnerFactory
    {
        private Tokenizer _tokenizer;

        private int _n_layer = 0;
        private int _n_embd = 0;

        private string _model;
        private OnnxModel _onnxModel;

        public RunnerFactory(string model, int n_layer, int n_embd)
        {
            _model = model;
            _n_layer = n_layer;
            _n_embd = n_embd;
        }

        public void Init()
        {
            TensorAllocator.InitDevices(ProcessorTypeEnums.CPU, new int[] { 0 });

            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
            _tokenizer = new Tokenizer(Path.Combine(path, "20B_tokenizer.json"));

            _onnxModel = new OnnxModel(Path.Combine(path, _model), _n_embd, _n_layer);
        }

        public Runner NewRunner()
        {
            return new Runner(_onnxModel, _tokenizer.NewEncoder());
        }
    }
}
