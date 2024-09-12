using RWKVSharp.Core.Sampler;
using RWKVSharp.Core.Tokenizer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Interfaces
{
    public interface IRunnerOptions
    {
        public Func<ITokenizer> Tokenizer { get; set; }
        public Func<ISampler> Sampler { get; set; }
    }
}
