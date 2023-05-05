using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CLLM.Core.Interfaces
{
    public interface IRunnerOptions
    {
        public ITokenizer? Tokenizer { get; set; }
        public int MaxTokens { get; set; }
    }
}
