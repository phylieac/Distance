/**
 * 数学函数库
 * @author pan
 */
package edu.pantek.deeplearning.math;

public class MathMethods {
	
	/**return the arc*/
	public double Max_acrcos(double dist1,double dist2 ){
		double acr;
		acr=Math.acos(dist1)+Math.acos(dist2);		
		return acr;
	}
	public double Min_acrcos(double dist1,double dist2){
         double acr;
         acr=Math.max(Math.acos(dist1), Math.acos(dist2))-Math.min(Math.acos(dist1), Math.acos(dist2));
         return acr;		
	}
	/**计算平均数*/
	public static double average(Float[] array){
		double aver=0;
		int size=array.length;
		for(int i=0;i<size;i++)
			aver+=array[i];
		aver=aver/size;
		return aver;
	}
	/**计算加权平均值*/
	public static double averaged(Float[] av,Float[] array){
		double aver=0;
		if (av.length==array.length){			
			for(int i=0;i<array.length;i++){
				aver+=av[i]*array[i];
			}
			aver=aver/array.length;
		}else
			System.out.println("权值与数值数不同，请检查权值！");
		return aver;
	}
	/**
	 * 计算标准差
	 * */
	public static double standardDeviation(Float[] array){
		double standar=0;
		double len=0;
		int size=array.length;
		standar=average(array);
		for(int i=0;i<size;i++){
			len+=(array[i]-standar)*(array[i]-standar);			
		}
		len/=size;
		standar=Math.sqrt(len);
		return standar;
	}
	/**
	 * 偏度
	 * */
	public static double skewness(Float[] array){
		double ske=average(array);
		double len=0;
		double low=0;	
		int size=array.length;
		for(int i=0;i<size;i++){
			len+=Math.pow(array[i]-ske,3);
			low+=Math.pow(array[i]-ske, 2);
		}
		ske=len*Math.pow(size, 1/2)/Math.pow(low, 3/2);
		return ske;
	}
	/***
	 * 计算平均数
	 * */
	public static double average(float[] array){
		double aver=0;
		int size=array.length;
		for(int i=0;i<size;i++)
			aver+=array[i];
		aver=aver/size;
		return aver;
	}
	/**计算加权平均值*/
	public static double averaged(float[] av,float[] array){
		double aver=0;
		if (av.length==array.length){			
			for(int i=0;i<array.length;i++){
				aver+=av[i]*array[i];
			}
			aver=aver/array.length;
		}else
			System.out.println("权值与数值数不同，请检查权值！");
		return aver;
	}
	/**计算标准差*/
	public static double standardDeviation(float[] array){
		double standar=0;
		double len=0;
		int size=array.length;
		standar=average(array);
		for(int i=0;i<size;i++){
			len+=(array[i]-standar)*(array[i]-standar);			
		}
		len/=size;
		standar=Math.sqrt(len);
		return standar;
	}
	/**偏度*/
	public static double skewness(float[] array){
		double ske=average(array);
		double len=0;
		int size=array.length;
		for(int i=0;i<size;i++){
			len+=Math.pow(array[i]-ske,3);
		}
		ske=len/((size-1)*(size-2)* standardDeviation(array));
		
		return ske;
	}
	public static void main(String [] args){
	
//		String word1="{（中国，美国），0.675158}";
//		String word2="{（美国，澳大利亚），0.675427}";
		MathMethods math=new MathMethods();
		double bigAcr= math.Max_acrcos(0.675158, 0.675427);
		double smallAcr=math.Min_acrcos(0.675158, 0.675427);
		System.out.println(Math.cos(bigAcr));
		System.out.println(Math.cos(smallAcr));
		System.out.println(Math.cos((bigAcr-smallAcr))/2);
		//System.out.println((int)(Math.PI/Math.acos(0.675158)));
		
	}
}
