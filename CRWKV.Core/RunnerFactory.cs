namespace RWKV
{
    public class RunnerFactory
    {
        private Tokenizer _tokenizer;

        public int N_layer { get; set; } = 0;
        public int N_embd { get; set; } = 0;
        public string ModelFile { get; set; }
        public string ModelPath { get; set; }
        public IRWKVModel Model { get; set; }

        public RunnerFactory(string model, int n_layer = 0, int n_embd = 0)
        {
            ModelFile = model;
            N_layer = n_layer;
            N_embd = n_embd;
        }

        public void Init()
        {
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Model");
            _tokenizer = new Tokenizer(Path.Combine(path, "20B_tokenizer.json"));
            ModelPath = path;
        }

        public Runner NewRunner()
        {
            return new Runner(Model, _tokenizer.NewEncoder());
        }
    }
}
