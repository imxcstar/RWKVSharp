using System.Text.Json;

namespace RWKV
{
    public class TokenizerInfo
    {
        public string version { get; set; }
        public object truncation { get; set; }
        public object padding { get; set; }
        public Added_Tokens[] added_tokens { get; set; }
        public Normalizer normalizer { get; set; }
        public Pre_Tokenizer pre_tokenizer { get; set; }
        public Post_Processor post_processor { get; set; }
        public Decoder decoder { get; set; }
        public Model model { get; set; }
    }

    public class Normalizer
    {
        public string type { get; set; }
    }

    public class Pre_Tokenizer
    {
        public string type { get; set; }
        public bool add_prefix_space { get; set; }
        public bool trim_offsets { get; set; }
    }

    public class Post_Processor
    {
        public string type { get; set; }
        public bool add_prefix_space { get; set; }
        public bool trim_offsets { get; set; }
    }

    public class Decoder
    {
        public string type { get; set; }
        public bool add_prefix_space { get; set; }
        public bool trim_offsets { get; set; }
    }

    public class Model
    {
        public string type { get; set; }
        public object dropout { get; set; }
        public object unk_token { get; set; }
        public object continuing_subword_prefix { get; set; }
        public object end_of_word_suffix { get; set; }
        public bool fuse_unk { get; set; }
        public Dictionary<string, int> vocab { get; set; }
        public string[] merges { get; set; }
    }


    public class Added_Tokens
    {
        public int id { get; set; }
        public bool special { get; set; }
        public string content { get; set; }
        public bool single_word { get; set; }
        public bool lstrip { get; set; }
        public bool rstrip { get; set; }
        public bool normalized { get; set; }
    }

    public class Tokenizer
    {
        public BPEncoder Encoder { get; set; }

        public Tokenizer(string path)
        {
            using var file = File.OpenRead(path);
            var data = JsonSerializer.Deserialize<TokenizerInfo>(file);
            if (data == null)
                throw new NotSupportedException();
            Encoder = new BPEncoder(data);
        }
    }
}
