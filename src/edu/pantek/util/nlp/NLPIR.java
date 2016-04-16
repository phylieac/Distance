package edu.pantek.util.nlp;
/**
 * NLPIR方法二次封装
 * @author Pan
 */
public class NLPIR {
	//初始化状态
	public static boolean InitState=false;
	//初始化方法
	public static boolean Init(String argu){
	    int	charsetType=1;
		boolean init_flag=false;
		try{
		    init_flag=CLibraryNLPIR.Instance .NLPIR_Init(argu,charsetType,"0");
		    //System.out.println(CLibraryNLPIR.Instance.NLPIR_GetLastErrorMsg() );
		}catch(Exception e){
			System.out.println("NLPIR参数解析失败！");
		}
		if(!init_flag){
			System.out.println("NLPIR 分词初始化失败！");
			return false;
		}
		else{
			InitState=true;
			return true;
		}
	}
	public static boolean Init(){		
		int charsetType=1;
		String  argu="./lib";
		boolean init_flag=false;
		try{
		    init_flag=CLibraryNLPIR.Instance .NLPIR_Init(argu,charsetType,"0");
		}catch(Exception e){
			System.out.println("NLPIR参数解析失败");
		}
		if(!init_flag){
			System.out.println("NLPIR 分词初始化失败！");
			return false;
		}else{
			InitState=true;
			return true;
		}
	}
	/**
	 * NLPIR分词方法
	 * @param sSrc 待分词字符串
	 * @param bPOSTagged 分词标注集序号
	 * 0----为无标注分词结果集，	 
	 * 1---- ICT_POS_MAP_FIRST  计算所一级标注集
	 * 2-----ICT_POS_MAP_SECOND  计算所二级标注集
	 * 3-----PKU_POS_MAP_SECOND   北大二级标注集	
	 *	4-----PKU_POS_MAP_FIRST 	  北大一级标注集
	 * @return
	 */
	public static String paragraphProcess(String sSrc, int bPOSTagged){
		String result="";
		if(!InitState){
			System.out.println("请先初始化分词！");
			return null;
		}
		try{
		result=CLibraryNLPIR.Instance.NLPIR_ParagraphProcess(sSrc, bPOSTagged);
		}catch(Exception e){
			System.out.println("参数解析失败");
		}
		return result;
	}
	/**
	 * 获取词的词性
	 * @param word 词
	 * @param bPOSTagged 标注集
	 * @return
	 */
	public static String getWordPos(String word,int bPOSTagged){
		if(bPOSTagged==0){
			System.out.println("bPOSTagged 不能为0！");
			return null;
		}			
		String pos="";
        String[] result;
		if(!InitState){
			System.out.println("请先初始化分词！");
			return null;
		}
		try{
		    result=CLibraryNLPIR.Instance.NLPIR_ParagraphProcess(word, bPOSTagged).split("/");
		    //System.out.println(result.length);
		    pos=result[1].split("\\s")[0];
		}catch(Exception e){
			System.out.println("参数解析失败！");
		}
		return pos;	
	}
	public static String getWordPos(String str){
		String result="";
		String pos="";
		if(!InitState){
			System.out.println("请先初始化分词！");
			return null;
		}
		try{
		    pos=CLibraryNLPIR.Instance.NLPIR_GetWordPOS(str);
		    //System.out.println(pos);
		    if(!pos.contains("#")) return "un";
		    String[] p=pos.split("#");
		    if(p.length>1){
		    	for(int i=0;i<p.length;i++){
		    		if(i==0) result+=p[i].split("/")[1]+"#";
		    		else if(i==p.length-1) result+=p[i].split("/")[1];
		    		else result+=p[i].split("/")[1]+"#";
		    	}
		    	return result;
		    }else{
		    	result=pos.split("/")[1];
		    }
		}catch(Exception e){
			System.out.println("参数解析失败！");
		}
		return result;	
	}
	/**
	 * 添加用户词典
	 * @param fileDict
	 */
	public static void importUserDict(String fileDict){
		if(!InitState){
			System.out.println("请先初始化分词！");
			return ;
		}
		CLibraryNLPIR.Instance.NLPIR_ImportUserDict(fileDict,true);
	}
	/**
	 * 添加用户词
	 * @param word
	 * @return
	 */
	public static boolean addUserWord(String word){
		if(!InitState){
			System.out.println("请先初始化分词！");
			return false ;
		}
		int addState= CLibraryNLPIR.Instance.NLPIR_AddUserWord(word);
		if(addState==0)
			return false;
		else 
			return true;
	}
	/**
	 * 保存用户词典
	 */
	public static void saveUserWord(){
		CLibraryNLPIR.Instance.NLPIR_SaveTheUsrDic();
	}
	
	public static boolean deleteUserWord(String word){
		if(!InitState){
			System.out.println("请先初始化分词！");
			return false ;
		}
		int delState=CLibraryNLPIR.Instance.NLPIR_DelUsrWord(word);
		if(delState==-1){
			System.out.print("The word :"+word+"not exsit!");
			return false;
		}else
			return true;
	}
	/**
	 * 获取关键词
	 * @param line
	 * @param MaxKeyLimit
	 * @param WeightOut
	 * @return
	 */
	public static String getKeyWords(String line, int MaxKeyLimit,boolean WeightOut){
		if(!InitState){
			System.out.println("请先初始化分词！");
			return null ;
		}
		String result=CLibraryNLPIR.Instance.NLPIR_GetKeyWords(line, MaxKeyLimit, WeightOut);
		
		return result;
	}
	/**
	 * 新词发现
	 * @param sLine
	 * @param nMaxKeyLimit
	 * @param bWeightOut
	 * @return
	 */
	public static String getNewWords(String sLine,int nMaxKeyLimit,boolean bWeightOut){
		if(!InitState){
			System.out.println("请先初始化分词！");
			return null ;
		}
		String result=CLibraryNLPIR.Instance.NLPIR_GetNewWords(sLine, nMaxKeyLimit, bWeightOut);
		return result;
	}
	public static void Exit(){
		if(!InitState){
			System.out.println("未初始化分词！无需执行退出操作!");
		}
		CLibraryNLPIR.Instance.NLPIR_Exit();
	}
	
	public static void main(String[] args) throws Exception {
		NLPIR.Init();
		//String str="Traditional Chinese Medicine anticancer injection Kanglaite, of which China has proprietary intellectual property right, has been approved by U.S. Food and Drug Administration to enter the phase III clinical trial and will expand usage among cancer patients in the US, according to news released by Zhejiang Chinese Medical University.";
		String str1="被不起诉人常学斌，男，1970年4月23日生，身份证号码320923197004230314，汉族，初中文化，驾驶员，住阜宁县东沟镇新东居委会条龙新村23号。被不起诉人常学斌因涉嫌交通肇事罪，于2014年1月15日被阜宁县公安局取保候审，同年2月24日经本院决定取保候审，当日由阜宁县公安局执行取保候审。 ";
		
		String result=NLPIR.paragraphProcess(str1, 1);
		System.out.println(result);
	}
}

