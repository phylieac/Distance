/**
 * @author Hongyan Pan----Phy
 * Date:2014.9.27
 * Main Calculation Class: Get N Words that Most Closest to the Given Word.
 * Date:2014.10.5  Added Object File Save as cache.
 * Date:2014.10.6  Added k-means method to cluster all vectors to n classes.
 * Date:2015.3  Present the POS-CBOW Model and POS-Skip-gram Model
 */
package edu.pantek.deeplearning.word2vec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.pantek.deeplearning.math.MathMethods;
import edu.pantek.util.nlp.NLPIR;

public class Distance {

	/**
	 * 参数设置
	 * */
	 private  long  N=40;                   //最优词数     number of closest words
	 private String vec_src;
	/**
	 * 内存参数声明
	 * */
	 private  long  vocabSize;
	 private  long  layerSize;
	 private  String [] best_word;
	 private  Float[] bestd;
	 Map<String,float[]> vecMap=new LinkedHashMap<String,float[]>();
	 int[] posTable=new int[26];
	 
	 /**
	  * 参数get和set方法
	  * */
	 private void set_N(long n){
		 this.N=n;
	 }
	 private void set_vec_file(String file){
		 this.vec_src=file;
	 }
     private void set_vocabSize(long size){
    	 this.vocabSize=size;
     }
     private long get_layerSize(){
    	 return this.layerSize;
     }
	 private void set_layerSize(long size){
		 this.layerSize=size;
	 }
	/**
	 * 读取二进制词向量文件
	 * Object File
	 * */
	@SuppressWarnings("unchecked")
	private void readBinaryVec(String Cache_file_path){
		 FileInputStream fi=null;
		 ObjectInputStream oi=null;		 
		 try {
			fi=new FileInputStream(Cache_file_path);
			oi=new ObjectInputStream(fi);			
			vecMap=  (Map<String, float[]>) oi.readObject();
		} catch (Exception e) {
			e.printStackTrace();
		}finally{
			try {
				oi.close();
				fi.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		 vocabSize=vecMap.size();
		 layerSize=vecMap.get("</s>").length;
	}
	/**
	 * 
	 * @param vector_file_path
	 */
	 private void readVector(String vector_file_path){
		 set_vec_file(vector_file_path);
		 File file =new File(vec_src);
		 if(file.exists()&&file.canRead()){
		 try{
		          FileReader reader=new FileReader(file);
		          BufferedReader buffer=new BufferedReader(reader);
		          String line=null;
		          String[] firstLine= buffer.readLine().split("\\s");
		          set_vocabSize(Integer.valueOf(firstLine[0]).intValue());
		          set_layerSize(Integer.valueOf(firstLine[1]).intValue());
		          String[] wordVec=new String[(int)get_layerSize()+1];
		         
		          while((line=buffer.readLine())!=null){ 
		        	           float[] vector = new  float[(int)get_layerSize()];
		                       wordVec=line.split("\\s");			    
			                   for(int i=1;i<wordVec.length;i++){				 
				                      vector[i-1]=Float.valueOf(wordVec[i]).floatValue();
			                   }		
			                  double len=0;
			                  for(int j=0;j<vector.length;j++){
				                        len+=vector[j]*vector[j];
			                   }
			                  len=Math.sqrt(len);
			                 for(int h=0;h<vector.length;h++){
				                        vector[h]=vector[h]/(float)len;
			                  }
			                 vecMap.put(wordVec[0], vector);			                
		 }		          
		 buffer.close();
		 reader.close();
		 }catch(IOException e){
			 System.out.println(e);
		 }
		 }else{
			 System.out.println("文件不可读或不存在！");
		 }
	}
   /**
	* 初始化词性表
	*/
	public void initPOSTable(){
		 int a=97;
		 for(int i=0;i<26;i++) posTable[i]=a+i;
	}		
	/**
	 *使用默认参数加载词向量文件 
	 * */
	public Distance(String filePath,boolean loadcache){
		this.initPOSTable();
		this.best_word=new String[(int)N];
		this.bestd=new Float[(int)N];
		if(!filePath.trim().isEmpty()){
			if(loadcache)
				     this.readBinaryVec(filePath);
			else
				     this.readVector(filePath);
		}
		else 
			System.out.println("词向量文件不存在或路径不正确！");
	}
	/**
	 * 设置相近词向量个数
	 * */
	public Distance(int cn,String filePath,boolean loadcache){
		this.set_N(cn);
		this.best_word=new String[(int)N];
		this.bestd=new Float[(int)N];
		this.initPOSTable();
		if(!filePath.trim().isEmpty()){
			if(loadcache)
				this.readBinaryVec(filePath);
		     else
			     this.readVector(filePath);
		}
        else 
	          System.out.println("词向量文件不存在或路径不正确！");
	}
   public int getPOS(String pos){
	   for(int x:posTable){
		   if(pos.indexOf(x)!=-1) return x;
	   }
	   return -1;
   }
	/**
	 * 计算词向量
	 * */
	public float calculateWords(String word1,String word2){
		float dist=0;
		if(!word1.trim().equals("")&&!word2.trim().equals("")){
		            float[] word1vec=vecMap.get(word1);
		            float[] word2vec=vecMap.get(word2);
		            if(word1vec!=null&&word2vec!=null){
			                float len=0;
			                 for(int i=0;i<word1vec.length;i++){
			                	 len+=word1vec[i]*word1vec[i];
			                 }
			                 len=(float)Math.sqrt(len);
		            	     for(int i=0;i<word1vec.length;i++){
		            	    	 word1vec[i]/=len;
		            	     }
		            	     if(word1vec.length==word2vec.length){
		            	         for(int i=0;i<word1vec.length;i++){
		            	        	       dist=dist+word1vec[i]*word2vec[i];
		            	         }
		            	     }else
		            	    	 System.out.println("检查\""+word1+"\"与\""+word2+"\" 的向量值！");
		            }else if(word1vec==null) 
                              System.out.println(word1+": Out of vocab");
		            else{
			                 System.out.println(word2+": Out of vocab");
		             }
		}else
			       System.out.println("参数不能为空字符！");
        return dist;
	}
	/**
	 * 计算word相近词Distance集合
	 * */
	public void calculateDistance(String word){
		       word=word.trim();
	           if(word.equals("")){
	        	   System.out.println("计算词不能为空！");
	        	   return;
	           }
	  	         for(int l=0;l<N;l++)   { 
	  	        	 bestd[l]=(float) -1;
	  	        	 best_word[l]=null;
	  	        }
	           float[] wordvec=vecMap.get(word);
	           if(wordvec!=null){
	        	   float len=0;
	        	   for(int i=0;i<wordvec.length;i++){
	        		   len+=wordvec[i]*wordvec[i];
	        	   }
	        	   len=(float)Math.sqrt(len);
	        	   for(int j=0;j<wordvec.length;j++){
	        		   wordvec[j]=wordvec[j]/len;
	        	   }
		           Iterator<Entry<String, float[]>> iter=vecMap.entrySet().iterator();
		           Map.Entry<String, float[]> entry;
		           float[] f;	     
		           String vectorWord;
		           while(iter.hasNext()){
		        	         float dist=0;
		                     entry=iter.next();
		                     vectorWord=entry.getKey();
		                     if (vectorWord.equals(word)) continue;
		        	         f=entry.getValue();	        	         	        	         
		        	         if(wordvec.length==f.length){	        	          	 
		        	        	 for(int k=0;k<wordvec.length;k++){
		        	        		 dist+=wordvec[k]*f[k];
		        	        	 }	        	        
		        	         }else
		        	        	 System.out.println("词 "+word+" ，"+entry.getKey()+"向量读取错误");
		        	         for(int m=0;m<N;m++){
		        	        	 if(dist>bestd[m]){
		        	        		 for(int d=(int)N-1;d>m;d--){
		        	        			 bestd[d]=bestd[d-1];
		        	        			 best_word[d]=best_word[d-1];
		        	        		 }
		        	        		 bestd[m]=dist;
		        	        		 best_word[m]=vectorWord;
		        	        		 break;
		        	        	 }
		        	         }	        	         
		           }
	           }else{
	        	   System.out.println("Word: "+word+" , out of vocab");
	           }	           
	}
	
	 public void calculateDistanceWithPOS(String wordPos){
		if(!NLPIR.InitState) NLPIR.Init();
		wordPos=wordPos.trim();
       if(wordPos.equals("")){
    	   System.out.println("计算词不能为空！");
    	   return;
       }
       String posInfo=NLPIR.getWordPos(wordPos);
       wordPos=wordPos+"/"+posInfo;
       int posNum=this.getPOS(posInfo);
	    for(int l=0;l<N;l++){
	        bestd[l]=(float) -1;
	        best_word[l]=null;
	    }
       float[] wordvec=vecMap.get(wordPos);
       if(wordvec!=null){
    	   float len=0;
    	   for(int i=0;i<wordvec.length;i++){
    		   len+=wordvec[i]*wordvec[i];
    	   }
    	   len=(float)Math.sqrt(len);
    	   for(int j=0;j<wordvec.length;j++){
    		   wordvec[j]=wordvec[j]/len;
    	   }
	           Iterator<Entry<String, float[]>> iter=vecMap.entrySet().iterator();
	           Map.Entry<String, float[]> entry;
	           float[] f;	     
	           String vectorWord;
	           while(iter.hasNext()){
	        	         float dist=0;
	                     entry=iter.next();
	                     vectorWord=entry.getKey();
	                     if (vectorWord.equals(wordPos)) continue;
	                     if(!vectorWord.equals("</s>")){
	                    	 if(vectorWord.split("/").length<=1) continue;
		                     if(posNum!=this.getPOS(vectorWord.split("/")[1])) continue; //&&this.getPOS(vectorWord.split("/")[1])!=-1
		                     if(vectorWord.split("/")[1].equals("un")) continue;
	                     }	                     
	        	         f=entry.getValue();
	        	         if(wordvec.length==f.length){
	        	        	 for(int k=0;k<wordvec.length;k++){
	        	        		 dist+=wordvec[k]*f[k];
	        	        	 }
	        	         }else
	        	        	 System.out.println("词 "+wordPos+" ，"+entry.getKey()+"向量读取错误");
	        	         for(int m=0;m<N;m++){
	        	        	 if(dist>bestd[m]){
	        	        		 for(int d=(int)N-1;d>m;d--){
	        	        			 bestd[d]=bestd[d-1];
	        	        			 best_word[d]=best_word[d-1];
	        	        		 }
	        	        		 bestd[m]=dist;
	        	        		 best_word[m]=vectorWord;
	        	        		 break;
	        	        	 }
	        	         }
	           }
       }else{
    	   System.out.println("Word: "+wordPos+" , out of vocab");
       }
}
	
	public void calculateDistanceByPOS(String wordPos){
		if(!NLPIR.InitState) NLPIR.Init();
		wordPos=wordPos.trim();
        if(wordPos.equals("")){
     	   System.out.println("计算词不能为空！");
     	   return;
        }
        String posInfo=NLPIR.getWordPos(wordPos);
        wordPos=wordPos+"/"+posInfo;
        int posNum=this.getPOS(posInfo);
        System.out.println("pos : "+(char)posNum);
	    for(int l=0;l<N;l++){
	        	 bestd[l]=(float) -1;
	        	 best_word[l]=null;
	    }
        float[] wordvec=vecMap.get(wordPos);
        if(wordvec!=null){
     	   float len=0;
     	   for(int i=0;i<wordvec.length;i++){
     		   len+=wordvec[i]*wordvec[i];
     	   }
     	   len=(float)Math.sqrt(len);
     	   for(int j=0;j<wordvec.length;j++){
     		   wordvec[j]=wordvec[j]/len;
     	   }
	           Iterator<Entry<String, float[]>> iter=vecMap.entrySet().iterator();
	           Map.Entry<String, float[]> entry;
	           float[] f;	     
	           String vectorWord;
	           while(iter.hasNext()){
	        	         float dist=0;
	                     entry=iter.next();
	                     vectorWord=entry.getKey();
	                     if (vectorWord.equals(wordPos)) continue;
	                     if(!vectorWord.equals("</s>")){
		                     if(posNum!=this.getPOS(vectorWord.split("/")[1])&&this.getPOS(vectorWord.split("/")[1])!=-1) continue;
	                     }
	        	         f=entry.getValue();	        	         	        	         
	        	         if(wordvec.length==f.length){	        	          	 
	        	        	 for(int k=0;k<wordvec.length;k++){
	        	        		 dist+=wordvec[k]*f[k];
	        	        	 }	        	        
	        	         }else
	        	        	 System.out.println("词 "+wordPos+" ，"+entry.getKey()+"向量读取错误");
	        	         for(int m=0;m<N;m++){
	        	        	 if(dist>bestd[m]){
	        	        		 for(int d=(int)N-1;d>m;d--){
	        	        			 bestd[d]=bestd[d-1];
	        	        			 best_word[d]=best_word[d-1];
	        	        		 }
	        	        		 bestd[m]=dist;
	        	        		 best_word[m]=vectorWord;
	        	        		 break;
	        	        	 }
	        	         }
	           }
        }else{
     	   System.out.println("Word: "+wordPos+" , out of vocab");
        }
}
	/**
	 * 计算大于distValue的word相近词Distance集合
	 * */
	public Map<String,Float> calculateDistance(float distValue,String word){
		       word=word.trim();
	           if(word.equals("")){
	        	   System.out.println("计算词不能为空！");	        	   
	        	   return null;
	           }	           
	  	         for(int l=0;l<N;l++)   { 
	  	        	 bestd[l]=(float) -1;
	  	        	 best_word[l]=null;
	  	        }
	  	       Map<String,Float>  wordVector=new LinkedHashMap<String,Float>();
	           float[] wordvec=vecMap.get(word);	          
	           if(wordvec!=null){
	        	   float len=0;
	        	   for(int i=0;i<wordvec.length;i++){
	        		   len+=wordvec[i]*wordvec[i];
	        	   }
	        	   len=(float)Math.sqrt(len);
	        	   for(int j=0;j<wordvec.length;j++){
	        		   wordvec[j]=wordvec[j]/len;
	        	   }
		           Iterator<Entry<String, float[]>> iter=vecMap.entrySet().iterator();
		           Map.Entry<String, float[]> entry;
		           float[] f;	     
		           String vectorWord;
		           List<String> wordList=new ArrayList<String>();
		           List<Float>  vectorList=new ArrayList<Float>();
		           while(iter.hasNext()){
		        	         float dist=0;
		                     entry=iter.next();
		                     vectorWord=entry.getKey();
		                     if (vectorWord.equals(word)) continue;
		                     
		        	         f=entry.getValue();	        	         	        	         
		        	         if(wordvec.length==f.length){	        	          	 
		        	        	 for(int k=0;k<wordvec.length;k++){
		        	        		 dist+=wordvec[k]*f[k];
		        	        	 }	        	        
		        	         }else{
		        	        	 System.out.println("词 "+word+" ，"+entry.getKey()+"向量读取错误");
		        	         }
		        	         if(dist>distValue){
	                                 wordList.add(vectorWord);
		        	        	     vectorList.add(dist);	        	        	 
		        	        }	        	          
		           }
		           int aSize=0;
		           if(wordList.size()==vectorList.size()) {
		                  aSize=wordList.size();
		           }else  
		        	   System.out.println("词向量分配出错！");
		           String[] wordsort=new String[aSize];
		           float[]  vectorsort=new float[aSize];
		           for(int i=0;i<aSize;i++)     vectorsort[i]=(float)-1;
		           float dist=0;
		           String sword;
		           for(int i=0;i<aSize;i++){
		        	   dist= vectorList.get(i);
		        	   sword=wordList.get(i);
		        	   for(int h=0;h<aSize;h++){
			                if(dist>vectorsort[h]){
	      	        		         for(int d=(int)aSize-1;d>h;d--){
	      	        			             vectorsort[d]=vectorsort[d-1];
	      	        			             wordsort[d]=wordsort[d-1];
	      	        		          }
	      	        		          vectorsort[h]=dist;
	      	        		          wordsort[h]=sword;
	      	        		         break;     	        	 
			        	    }
			            }		            
		           }
		           for(int i=0;i<aSize;i++){
		        	   wordVector.put(wordsort[i], vectorsort[i]);
		           }
		           wordList.clear();
		           vectorList.clear();
		           return wordVector;
	           }else{
	        	   System.out.println("Word: "+word+" , out of vocab");
	        	   return null;
	           }
	}
	/**
	 * 向量计算  word1-word2+word3=word4;  word1与word2的关系,推测word3与word4的关系.
	 * */
	public void distanceAnalogy(String word1,String word2,String word3){
		word1=word1.trim();
		word2=word2.trim();
		word3=word3.trim();
		if(word1.equals("")){			 
			     System.out.println("输入词不能为空！");
			     return;
		}
		if(word2.equals("")){
	    	     System.out.println("输入词不能为空！");
                 return;
	     }
   	    if(word3.equals("")){
		         System.out.println("输入词不能为空！");
		         return;
	     }
		float[] f1=vecMap.get(word1);
		float[] f2=vecMap.get(word2);
		float[] f3=vecMap.get(word3);
		if(f1==null){
			System.out.println("Word: "+word1+" , out of vocab");
			return;
		}if(f2==null){
			System.out.println("Word: "+word2+" , out of vocab");
			return;
		}if(f3==null){
			System.out.println("Word: "+word3+" , out of vocab");
			return;
		}
	     for(int l=0;l<N;l++)   { 
	        	 bestd[l]=(float) -1;
	        	 best_word[l]=null;
	      }

		 if(f1.length!=f2.length&&f1.length!=f3.length){
			 System.out.println("词向量读取出错！");
		     return;
		}
		float[] wordvec=new float[(int)layerSize];
		for(int i=0;i<(int)layerSize;i++)    wordvec[i]=f2[i]-f1[i]+f3[i];
      	float len=0;
      	for(int i=0;i<wordvec.length;i++){
      		   len+=wordvec[i]*wordvec[i];
      	 }
      	len=(float)Math.sqrt(len);
      	for(int j=0;j<wordvec.length;j++){
      		   wordvec[j]=wordvec[j]/len;
      	}
         Iterator<Entry<String, float[]>> iter=vecMap.entrySet().iterator();
         Map.Entry<String, float[]> entry;
         float[] f;	     
         String vectorWord;
         while(iter.hasNext()){
      	         float dist=0;
                 entry=iter.next();
                 vectorWord=entry.getKey();
                 if (vectorWord.equals(word1)) continue;
                 if (vectorWord.equals(word2)) continue;
                 if (vectorWord.equals(word3)) continue;
      	         f=entry.getValue();	        	         	        	         
      	         if(wordvec.length==f.length){	        	          	 
      	        	 for(int k=0;k<wordvec.length;k++){
      	        		 dist+=wordvec[k]*f[k];
      	        	 }	        	        
      	         }else
      	        	 System.out.println("词 "+word1+" ，"+word2+" , "+word3+"  "+"向量计算错误！");
      	         for(int m=0;m<N;m++){
      	        	       if(dist>bestd[m]){
      	        		           for(int d=(int)N-1;d>m;d--){
      	        			               bestd[d]=bestd[d-1];
      	        			               best_word[d]=best_word[d-1];
      	        		           }
      	        		           bestd[m]=dist;
      	        		           best_word[m]=vectorWord;
      	        		           break;
      	        	       }
      	        	 }
      	         }	        	         
	}
    /**
     * 获取计算结果集合
     * @return
     */
	public Map<String,Float> getResultMap(){
		Map<String,Float> result=new LinkedHashMap<String,Float>();
		if(best_word[0]==null) return null;
		if(bestd.length==best_word.length){
			for(int i=0;i<N;i++){
				if(best_word[i]==null) break;
				result.put(best_word[i], bestd[i]);
			}
		}else{
			System.out.println("计算结果集出错！");
			return null;
		}
		return result;
	}
	/**
	 * 序列化Object
	 * @param cache_file
	 * @param o
	 */
	private void SaveObject(String cache_file,Object o){
		FileOutputStream fw = null;
		ObjectOutputStream ow=null;
		try {
			fw=new FileOutputStream(cache_file);
			ow=new ObjectOutputStream(fw);
			ow.writeObject(o);
		} catch (Exception e) {
			e.printStackTrace();
		}finally{
			try {
				ow.close();
				fw.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
	}
    public void SaveClasses(String Out_File,Map<String,Integer> result){
  	  FileWriter wr = null;
  	  BufferedWriter buff=null;
  	  String str;
  	  try {
			 wr=new FileWriter(Out_File,true);
			 buff=new BufferedWriter(wr);			
			 buff.flush();
			 Iterator<Entry<String, Integer>> it=result.entrySet().iterator();
			 Map.Entry<String, Integer> entry;
			 int i=0;
			 while(it.hasNext()){
				 entry=it.next();
				 str=entry.getKey()+" "+entry.getValue();
				 str+="\n";
				 buff.write(str);
				 i++;
				 if(i%1000==0) buff.flush();
			 }
		  } catch (IOException e) {
			    e.printStackTrace();
		   }finally{
			   try {
				   buff.close();
				   wr.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
   }
    /**
     * Map Sort利用Collections的sort函数排序 降序
     * @param map
     * @return
     */
    private Map<String,Integer> SortMapAsValue(Map<String,Integer> map){
    	ArrayList<Map.Entry<String, Integer>> list=new ArrayList<Map.Entry<String,Integer>>(map.entrySet());
    	Collections.sort(list,new Comparator<Map.Entry<String,Integer>>(){
    		public int compare(Entry<String,Integer> arg0,Entry<String,Integer> arg1){
    			return arg1.getValue()-arg0.getValue();
    		}
    	});
    	Map<String,Integer> result=new LinkedHashMap<String,Integer>();
    	for(int i=0;i<list.size();i++){
    		result.put(list.get(i).getKey(), list.get(i).getValue());
    	}
    	list.clear();
    	list=null;
    	return result;
    }
	/**
	 * 词向量K-Means聚类
	 */
	public void KMeans(String file,int classes,boolean save_text){
		int iter=10,closeid;
		int[] centcn=new int[classes];
		List<Integer> cl=new ArrayList<Integer>();
		float closev,x;
		List<Float[]> cent=new ArrayList<Float[]>();
		List<float[]> vec=new ArrayList<float[]>();
		for(int i=0;i<vocabSize;i++) cl.add(i%classes);
		for(int i=0;i<iter;i++){
			 System.out.println("第 "+(i+1)+" 次聚类迭代");
			for(int j=0;j<classes;j++) {
				Float[] v=new Float[(int)layerSize];
				for(int h=0;h<layerSize;h++) v[h]=(float)0;
				cent.add(v);
			}
			for(int j=0;j<classes;j++) centcn[j]=1;
			Iterator<Entry<String,float[]>> it=vecMap.entrySet().iterator();
			Map.Entry<String, float[]> entry;
			while(it.hasNext()){
				 entry=it.next();
				 vec.add(entry.getValue());
			}
             for(int k=0;k<vocabSize;k++){           	
            	for(int a=0;a<layerSize;a++){
            		cent.get(cl.get(k))[a]+=vec.get(k)[a];
            	}
            	centcn[cl.get(k)]++;
            }
             for(int k=0;k<classes;k++){
            	 closev=0;
            	 for(int a=0;a<layerSize;a++){
            		 cent.get(k)[a]/=centcn[k];
            		 closev+=cent.get(k)[a]*cent.get(k)[a];
            	 }
            	 closev=(float) Math.sqrt(closev);
            	 for(int a=0;a<layerSize;a++) cent.get(k)[a]/=closev;
             }
             for(int k=0;k<vocabSize;k++){
            	 closev=-10;
            	 closeid=0;
            	 for(int a=0;a<classes;a++){
            		 x=0;
            		 for(int b=0;b<layerSize;b++) x+=cent.get(a)[b]*vec.get(k)[b];
            		 if(x>closev){
            			 closev=x;
            			 closeid=a;
            		 }
            	 }
            	cl.set(k, closeid);           	
             }
             System.out.println("已完成: "+((float)i/iter)*100+"%");
		}
		Map<String,Integer> ClassesResult=new LinkedHashMap<String,Integer>();
		Iterator<Entry<String,float[]>> itor=vecMap.entrySet().iterator();
	    Map.Entry<String, float[]> entry1;
	    for(int i=0;i<vocabSize;i++){
	    	entry1=itor.next();
	    	ClassesResult.put(entry1.getKey(), cl.get(i));
	    }
	    ClassesResult= SortMapAsValue(ClassesResult);
	    if(save_text) SaveClasses(file,ClassesResult);
	    else SaveObject(file,ClassesResult);
	    System.out.println("Save Result :  "+System.getProperty("user.dir")+file.substring(1).replaceAll("/", "\\\\"));
	}
	
	/**
	 * 计算平均值
	 * */
	public  double averageResult(){
		double aver=0;
		if(bestd.length ==0){
			return (Double) 0.0;
		}else
			aver= MathMethods.average(bestd);
		return aver;		
	}
	/**
	 * 计算标准差
	 * */
	public  double standardDeviationResult(){
		double aver=0;
		if(bestd.length ==0){
			return (Double) 0.0;
		}else
			aver= MathMethods.standardDeviation(bestd);
		return aver;		
	}
	/**
	 * 计算偏度
	 * */
	public  double skewness(){
		double aver=0;
		if(bestd.length ==0){
			return aver;
		}else
			aver= MathMethods.skewness(bestd);
		return aver;		
	}
		
	public static void main(String[] args) throws IOException{
		String vecPath="./POS-CBOW.dat";
 		Distance dis=new Distance(20,vecPath,true);
		dis.calculateDistanceWithPOS("哲学");
		System.out.println(dis.getResultMap().toString());
	}	
} 